using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections;

public class UdpReceiver : MonoBehaviour
{
    // --- Public Fields (Link these in the Unity Inspector) ---
    [Header("Game Objects")]
    [SerializeField] private List<GameObject> balls;
    [SerializeField] private GameObject handMarker;
    [SerializeField] private GameObject levelCompleteBanner;

    [Header("Audio")]
    [SerializeField] private AudioSource completeSound;
    [SerializeField] private AudioSource ballBlastSound;

    [Header("Network Settings")]
    [SerializeField] private int port = 5053;

    [Header("Gameplay Settings")]
    [Tooltip("Smoothing factor for hand movement. 0 = no smoothing, 1 = instant snap.")]
    [Range(0.1f, 1.0f)]
    [SerializeField] private float smoothingAlpha = 0.6f;
    [Tooltip("Maximum age of a network packet in seconds before it's ignored.")]
    [SerializeField] private float maxPacketAge = 0.5f;

    // --- Private Fields ---
    private Thread udpReceiveThread;
    private UdpClient udpClient;
    private Vector2 smoothedHandPos = Vector2.zero;
    private List<Vector2> ballSpeeds = new List<Vector2>();
    private List<bool> ballIsBlasted;
    private int ballsRemaining;
    private Camera mainCamera;

    // Thread-safe way to pass data from the network thread to the main thread
    private volatile string lastReceivedPacket = "";
    private object lockObject = new object();

    // This class must match the JSON structure from your Python script
    [System.Serializable]
    private class HandData
    {
        public float hand_depth;
        public float projector_depth;
        public float[] hand_position;
        public double ts;
    }

    void Start()
    {
        mainCamera = Camera.main;

        // --- Initialize Game State (like Godot's _ready()) ---
        if (levelCompleteBanner != null) levelCompleteBanner.SetActive(false);
        if (handMarker != null) handMarker.SetActive(false);

        InitializeBalls();

        // --- Start Network Listener ---
        udpReceiveThread = new Thread(new ThreadStart(ReceiveData));
        udpReceiveThread.IsBackground = true;
        udpReceiveThread.Start();

        Debug.Log($"📡 Listening on UDP port {port}");
    }

    void Update()
    {
        // --- Run every frame (like Godot's _process()) ---
        MoveBalls(Time.deltaTime);
        ProcessUdpData();
    }

    private void InitializeBalls()
    {
        ballIsBlasted = new List<bool>();
        ballsRemaining = balls.Count;

        // Get screen boundaries in world coordinates
        Vector2 screenBounds = mainCamera.ScreenToWorldPoint(new Vector3(Screen.width, Screen.height, mainCamera.transform.position.z));

        for (int i = 0; i < balls.Count; i++)
        {
            ballIsBlasted.Add(false);
            if (balls[i] != null)
            {
                // Randomize position
                float randomX = Random.Range(-screenBounds.x * 0.9f, screenBounds.x * 0.9f);
                float randomY = Random.Range(-screenBounds.y * 0.9f, screenBounds.y * 0.9f);
                balls[i].transform.position = new Vector2(randomX, randomY);

                // Randomize speed
                float speedX = Random.Range(1f, 3f) * (Random.value > 0.5f ? 1 : -1);
                float speedY = Random.Range(1f, 3f) * (Random.value > 0.5f ? 1 : -1);
                ballSpeeds.Add(new Vector2(speedX, speedY));
            }
        }
    }

    private void MoveBalls(float deltaTime)
    {
        Vector2 screenBounds = mainCamera.ScreenToWorldPoint(new Vector3(Screen.width, Screen.height, mainCamera.transform.position.z));

        for (int i = 0; i < balls.Count; i++)
        {
            if (balls[i] == null || ballIsBlasted[i]) continue;

            Vector2 newPos = (Vector2)balls[i].transform.position + ballSpeeds[i] * deltaTime;

            // Wall collision checks
            float ballRadius = balls[i].transform.localScale.x / 2; // Simple radius estimate
            if (newPos.x > screenBounds.x - ballRadius || newPos.x < -screenBounds.x + ballRadius)
            {
                ballSpeeds[i] = new Vector2(-ballSpeeds[i].x, ballSpeeds[i].y);
            }
            if (newPos.y > screenBounds.y - ballRadius || newPos.y < -screenBounds.y + ballRadius)
            {
                ballSpeeds[i] = new Vector2(ballSpeeds[i].x, -ballSpeeds[i].y);
            }

            balls[i].transform.position = newPos;
        }
    }

    private void ProcessUdpData()
    {
        string packet;
        lock (lockObject)
        {
            packet = lastReceivedPacket;
            lastReceivedPacket = ""; // Clear after reading
        }

        if (string.IsNullOrEmpty(packet)) return;

        Debug.Log("1. Received Raw Packet: " + packet);

        // Parse the JSON packet
        HandData data = JsonUtility.FromJson<HandData>(packet);
        if (data == null || data.hand_position == null || data.hand_position.Length < 2) return;

        // Timestamp staleness check
        double now = (System.DateTime.UtcNow - new System.DateTime(1970, 1, 1)).TotalSeconds;
        if (System.Math.Abs(now - data.ts) > maxPacketAge)
        {
            Debug.LogError("Packet is too old or timestamp is missing! Marker will not be shown.");
            if (handMarker != null) handMarker.SetActive(false);
            return; // Packet is too old
        }

        // Depth difference check
        float diff = Mathf.Abs(data.projector_depth - data.hand_depth);
        if (diff > 1.0f)
        {
            if (handMarker != null) handMarker.SetActive(false);
            return;
        }

        if (handMarker != null) handMarker.SetActive(true);

        // Convert received coordinates to world coordinates
        // NOTE: You must adjust this logic based on how your Python script sends coordinates.
        // This example assumes normalized screen coordinates (0 to 1).
        Vector3 screenPos = new Vector3(data.hand_position[0] * Screen.width, data.hand_position[1] * Screen.height, mainCamera.nearClipPlane + 1);
        Vector2 rawHandPos = mainCamera.ScreenToWorldPoint(screenPos);
        Debug.Log($"4. Calculated World Position: {rawHandPos}");
        // Smooth the position
        smoothedHandPos = Vector2.Lerp(smoothedHandPos, rawHandPos, smoothingAlpha);

        if (handMarker != null)
        {
            handMarker.transform.position = smoothedHandPos;

            // Blast logic check
            for (int i = 0; i < balls.Count; i++)
            {
                if (balls[i] == null || ballIsBlasted[i]) continue;

                float distance = Vector2.Distance(handMarker.transform.position, balls[i].transform.position);
                float ballRadius = balls[i].transform.localScale.x / 2;
                if (distance < ballRadius && diff < 0.5f)
                {
                    Debug.Log($"💥 Hit detected with {balls[i].name}");
                    StartCoroutine(BlastBall(i));
                }
            }
        }
    }

    private IEnumerator BlastBall(int ballIndex)
    {
        ballIsBlasted[ballIndex] = true;
        GameObject ball = balls[ballIndex];

        if (ballBlastSound != null) ballBlastSound.Play();

        // Fade out animation
        SpriteRenderer sr = ball.GetComponent<SpriteRenderer>();
        Color originalColor = sr.color;
        for (float t = 0; t < 1.0f; t += Time.deltaTime * 2.0f)
        {
            sr.color = new Color(originalColor.r, originalColor.g, originalColor.b, Mathf.Lerp(1, 0, t));
            yield return null;
        }

        ball.SetActive(false);
        ballsRemaining--;
        CheckLevelComplete();
    }

    private void CheckLevelComplete()
    {
        if (ballsRemaining <= 0)
        {
            if (levelCompleteBanner != null) levelCompleteBanner.SetActive(true);
            if (completeSound != null) completeSound.Play();
            Debug.Log("🎉 Level Complete!");
        }
    }

    // --- Network Thread ---
    private void ReceiveData()
    {
        udpClient = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = udpClient.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                lock (lockObject)
                {
                    lastReceivedPacket = text;
                }
            }
            catch (System.Exception err)
            {
                Debug.LogError(err.ToString());
            }
        }
    }

    void OnDestroy()
    {
        // Important: Clean up the thread and client when the game stops
        if (udpReceiveThread != null)
        {
            udpReceiveThread.Abort();
        }
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}
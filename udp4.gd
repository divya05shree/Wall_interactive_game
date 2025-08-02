extends Node2D

var udp := PacketPeerUDP.new()
var PORT := 5053

var balls = []
var ball_speeds = []
@onready var status_label = $Label
@onready var hand_marker = $HandMarker
@onready var level_complete_banner = $LevelCompleteBanner
@onready var complete_sound = $CompleteSound  # ensure the node name matches
@onready var background = $Background

var screen_size = get_viewport_rect().size

# smoothing / prediction
var smoothed_hand_pos := Vector2.ZERO
var last_raw_hand_pos := Vector2.ZERO
var last_update_time := 0.0  # in seconds
var velocity := Vector2.ZERO
const MIN_ALPHA := 0.55  # more smoothing when slow
const MAX_ALPHA := 0.85  # less smoothing when fast

# timing / staleness
const MAX_PACKET_AGE := 0.5  # seconds
var level_completed = false

func _ready():
	level_complete_banner.visible = false
	hand_marker.visible = false
	udp.bind(PORT)
	screen_size = get_viewport_rect().size
	# Background scaling if needed; remove if undesired
	background.scale = Vector2(1920, 1080) / background.texture.get_size()

	print("🖥 Screen Size at Runtime:", screen_size)
	print("📡 Listening on UDP port", PORT)

	for i in range(1, 8):  # BALL1 to BALL7
		var ball = get_node_or_null("BALL" + str(i))
		if ball:
			balls.append(ball)
			var tex_size = ball.texture.get_size() * ball.scale
			ball.position = Vector2(
				randf_range(0, screen_size.x - tex_size.x),
				randf_range(0, screen_size.y - tex_size.y)
			)
			var speed_x = randf_range(-200, 200)
			if abs(speed_x) < 100:
				speed_x = sign(speed_x) * 100
			var speed_y = randf_range(-200, 200)
			if abs(speed_y) < 100:
				speed_y = sign(speed_y) * 100
			ball_speeds.append(Vector2(speed_x, speed_y))

func _process(delta):
	move_balls(delta)
	process_udp()

func move_balls(delta):
	for i in range(balls.size()):
		var ball = balls[i]
		if ball == null or ball.get_meta("blasted", false):
			continue

		var tex_size = ball.texture.get_size() * ball.scale
		var new_pos = ball.position + ball_speeds[i] * delta

		if new_pos.x <= 0:
			new_pos.x = 0
			ball_speeds[i].x *= -1
		elif new_pos.x >= screen_size.x - tex_size.x:
			new_pos.x = screen_size.x - tex_size.x
			ball_speeds[i].x *= -1

		if new_pos.y <= 0:
			new_pos.y = 0
			ball_speeds[i].y *= -1
		elif new_pos.y >= screen_size.y - tex_size.y:
			new_pos.y = screen_size.y - tex_size.y
			ball_speeds[i].y *= -1

		ball.position = new_pos

func process_udp():
	if udp.get_available_packet_count() == 0:
		return

	var packet = udp.get_packet().get_string_from_utf8()
	var parsed = JSON.parse_string(packet)
	if not parsed:
		return
	var json = parsed

	# Extract fields defensively
	var hand_depth = json.has("hand_depth") ? json["hand_depth"] : null
	var proj_depth = json.has("projector_depth") ? json["projector_depth"] : null
	var hand_pos_arr = json.has("hand_position") ? json["hand_position"] : null
	var ts = json.has("ts") ? json["ts"] : null

	if hand_depth == null or proj_depth == null or hand_pos_arr == null:
		return

	# Timestamp staleness check (use float seconds from Python)
	if ts != null:
		var now = OS.get_system_time_secs()
		if abs(float(now) - float(ts)) > MAX_PACKET_AGE:
			return  # stale

	var raw_hand_pos = Vector2(hand_pos_arr[0], hand_pos_arr[1])

	# Depth difference
	var diff = absf(proj_depth - hand_depth)

	# Depth-based visibility / early out
	if diff > 1.0:
		hand_marker.visible = false
		return
	else:
		hand_marker.visible = true

	# Time delta for velocity estimation
	var current_time = OS.get_system_time_secs()
	var dt = 0.0
	if last_update_time > 0.0:
		dt = current_time - last_update_time
	last_update_time = current_time
	if dt <= 0.001:
		dt = 0.001  # avoid extreme spikes

	# Velocity estimate and clamping
	velocity = (raw_hand_pos - last_raw_hand_pos) / dt
	last_raw_hand_pos = raw_hand_pos
	if velocity.length() > 1000:
		velocity = velocity.normalized() * 1000

	# Prediction (small factor so it doesn't overshoot)
	var predicted_pos = raw_hand_pos + velocity * 0.05

	# Adaptive smoothing alpha
	var speed = velocity.length()
	var adapt_alpha = clamp(lerp(MIN_ALPHA, MAX_ALPHA, min(speed / 200.0, 1.0)), MIN_ALPHA, MAX_ALPHA)

	# Very small movement: snap to raw
	if speed < 5:
		smoothed_hand_pos = raw_hand_pos
	else:
		if smoothed_hand_pos == Vector2.ZERO:
			smoothed_hand_pos = predicted_pos
		else:
			smoothed_hand_pos = smoothed_hand_pos.linear_interpolate(predicted_pos, adapt_alpha)

	# Update marker display
	status_label.text = "Diff: %.2f | Hand Pos: %s" % [diff, str(smoothed_hand_pos)]
	hand_marker.position = smoothed_hand_pos

	# Blast logic: require closer depth and overlap
	for ball in balls:
		if ball == null or ball.get_meta("blasted", false):
			continue
		var ball_rect = Rect2(ball.position, ball.texture.get_size() * ball.scale)
		if ball_rect.has_point(smoothed_hand_pos) and diff < 0.5:
			print("💥 Hit detected with ", ball.name)
			ball.set_meta("blasted", true)
			blast_ball(ball)

func blast_ball(ball: Sprite2D):
	ball.modulate = Color(1, 0, 0, 1)
	var tween = get_tree().create_tween()
	tween.tween_property(ball, "modulate:a", 0.0, 1.0)
	tween.tween_callback(Callable(ball, "queue_free"))
	await get_tree().create_timer(0.2).timeout
	if check_all_balls_blasted():
		show_level_complete_banner()

func check_all_balls_blasted():
	for ball in balls:
		if ball != null and not ball.get_meta("blasted", false):
			return false
	return true

func show_level_complete_banner():
	if level_completed:
		return
	level_completed = true

	level_complete_banner.visible = true
	level_complete_banner.position = Vector2(700, 350)
	complete_sound.play()

	var tween = get_tree().create_tween()
	tween.tween_property(level_complete_banner, "position", Vector2(700, 350), 0.5).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_OUT)

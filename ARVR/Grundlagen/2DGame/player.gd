extends Area2D
@export var speed = 400

signal hit


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	var velocity = Vector2.ZERO
	
	
	if(Input.is_action_pressed("DOWN")):
		velocity.y = 1

	if(Input.is_action_pressed("UP")):
		velocity.y = -1

	if(Input.is_action_pressed("LEFT")):
		velocity.x = -1

	if(Input.is_action_pressed("RIGHT")):
		velocity.x = 1

	if(velocity.length() != 0):
		get_node("AnimatedSprite2D").play()
		if(velocity.y != 0):
			get_node("AnimatedSprite2D").animation = "up"

		if(velocity.x != 0):
			get_node("AnimatedSprite2D").animation = "walk"
			get_node("AnimatedSprite2D").flip_h = velocity.x < 0
	else:
		get_node("AnimatedSprite2D").stop()
		

	position = position + velocity * delta * speed#postion is aus dem Node2D Transform
		
	pass


func _on_body_entered(body):
	print("etwas")
	hide()
	hit.emit()
	$CollisionShape2D.set_deferred("disabled", true)
	pass # Replace with function body.

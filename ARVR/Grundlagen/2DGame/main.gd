extends Node2D

@export var mob_scene:PackedScene

# Called when the node enters the scene tree for the first time.
func _ready():
	
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_timer_timeout():
	var mob = mob_scene.instantiate()
	var mob_location = $Path2D/PathFollow2D
	
	mob_location.progress_ratio = randf()
	mob.position = mob_location.position
	mob.rotation = mob_location.rotation + PI/2
	var velo = Vector2(randf_range(50,200),0)
	velo = velo.rotated(mob.rotation)
	mob.linear_velocity = velo
	add_child(mob)


func _on_player_hit():
	print("SPieler getroffen")
	$Mob.stop()
	pass # Replace with function body.

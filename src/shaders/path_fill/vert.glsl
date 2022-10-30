#version 330

in vec2 control_point;
in float control_point_flag;

out vec2 vert_control_point;
out float vert_control_point_flag;

void main() {
	vert_control_point = control_point;
	vert_control_point_flag = control_point_flag;
}

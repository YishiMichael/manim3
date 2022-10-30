#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2 vert_control_point[3];
in float vert_control_point_flag[3];

out vec2 geom_point;
out vec2 geom_control_point_0;
out vec2 geom_control_point_1;
out vec2 geom_control_point_2;
out float geom_control_point_flag;

void main() {
	geom_control_point_0 = vert_control_point[0];
	geom_control_point_1 = vert_control_point[1];
	geom_control_point_2 = vert_control_point[2];
	geom_control_point_flag = vert_control_point_flag[1];

	for (int i = 0; i < 3; ++i) {
		geom_point = vert_control_point[i];
		gl_Position = vec4(geom_point, 0.0, 1.0);
		EmitVertex();
	}
	EndPrimitive();
}

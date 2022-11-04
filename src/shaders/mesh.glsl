#version 330

#if defined VERTEX_SHADER

uniform mat4 uniform_transform_matrix;

in vec3 in_position;

void main() {
	gl_Position = uniform_transform_matrix * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

uniform vec4 uniform_color;

out vec4 frag_color;

void main() {
	frag_color = uniform_color;
}

#endif

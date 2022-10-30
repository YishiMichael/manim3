#version 330

in vec2 geom_point;
in vec2 geom_control_point_0;
in vec2 geom_control_point_1;
in vec2 geom_control_point_2;
in float geom_control_point_flag;

out vec4 frag_color;

float cross2d(vec2 v, vec2 w) {
	return v.x * w.y - w.x * v.y;
}

float bezier_ratio_from_control_point() {
	// Let a quadratic bezier curve be determined by P0, P1, P2.
	// Let the line PP1 interset with the curve at S and line P0P2 at T.
	// Returns the ratio P1Q / P1S.

	// Transform P1 to the origin.
	vec2 p0 = geom_control_point_0 - geom_control_point_1;
	vec2 p2 = geom_control_point_2 - geom_control_point_1;
	vec2 p = geom_point - geom_control_point_1;

	// `t_ratio` describes the position of T.
	// It's -1 and 1 when T overlaps with P0 and P2, respectively.
	float t_ratio = cross2d(p0 + p2, p) / cross2d(p0 - p2, p);
	// P1T / P1S
	float t_over_s = t_ratio == 0.0 ? 2.0 : (t_ratio * t_ratio) / (1 - sqrt(1 - t_ratio * t_ratio));
	// P1Q / P1T
	float q_over_t = cross2d(p0 - p2, p) / cross2d(p0, p2);
	return abs(t_over_s * q_over_t);  // TODO: tackle this abs
}

bool point_in_frag() {
	if (geom_control_point_flag == 0.0) {
		return true;
	}
	float ratio = bezier_ratio_from_control_point();
	return geom_control_point_flag > 0.0 ^^ ratio < 1.0;
}

void main() {
	frag_color = vec4(1.0, 1.0, 1.0, 1.0);
	if (!point_in_frag()) {
		discard;
	}
}

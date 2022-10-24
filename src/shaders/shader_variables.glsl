//// background_frag.glsl ////

uniform sampler2D t2D;
varying vec2 vUv;


//// background_vert.glsl ////

varying vec2 vUv;
uniform mat3 uvTransform;


//// cube_frag.glsl ////

#if ( defined( USE_ENVMAP ) )
	uniform float envMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float flipEnvMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENVMAP_TYPE_CUBE ) )
	uniform samplerCube envMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENVMAP_TYPE_CUBE ) )
	uniform sampler2D envMap;
#endif
uniform float opacity;
varying vec3 vWorldDirection;


//// cube_vert.glsl ////

varying vec3 vWorldDirection;


//// depth_frag.glsl ////

#if ( DEPTH_PACKING == 3200 )
	uniform float opacity;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif
varying vec2 vHighPrecisionZW;


//// depth_vert.glsl ////

#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
varying vec2 vHighPrecisionZW;


//// distanceRGBA_frag.glsl ////

uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// distanceRGBA_vert.glsl ////

varying vec3 vWorldPosition;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// equirect_frag.glsl ////

uniform sampler2D tEquirect;
varying vec3 vWorldDirection;


//// equirect_vert.glsl ////

varying vec3 vWorldDirection;


//// linedashed_frag.glsl ////

uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// linedashed_vert.glsl ////

uniform float scale;
varying float vLineDistance;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshbasic_frag.glsl ////

uniform vec3 diffuse;
uniform float opacity;
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_AOMAP ) )
	uniform sampler2D aoMap;
#endif
#if ( defined( USE_AOMAP ) )
	uniform float aoMapIntensity;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform sampler2D lightMap;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform float lightMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float envMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float flipEnvMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENVMAP_TYPE_CUBE ) )
	uniform samplerCube envMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENVMAP_TYPE_CUBE ) )
	uniform sampler2D envMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float reflectivity;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
#if ( defined( USE_SPECULARMAP ) )
	uniform sampler2D specularMap;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshbasic_vert.glsl ////

#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	uniform mat3 uv2Transform;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshlambert_frag.glsl ////

uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_AOMAP ) )
	uniform sampler2D aoMap;
#endif
#if ( defined( USE_AOMAP ) )
	uniform float aoMapIntensity;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform sampler2D lightMap;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform float lightMapIntensity;
#endif
#if ( defined( USE_EMISSIVEMAP ) )
	uniform sampler2D emissiveMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float envMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float flipEnvMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENVMAP_TYPE_CUBE ) )
	uniform samplerCube envMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENVMAP_TYPE_CUBE ) )
	uniform sampler2D envMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float reflectivity;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
#if ( NUM_DIR_LIGHTS > 0 )
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif
#if ( NUM_POINT_LIGHTS > 0 )
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHTS > 0 )
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_1; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_2; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if ( NUM_HEMI_LIGHTS > 0 )
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
varying vec3 vViewPosition;
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_MAPS > 0 )
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_SPECULARMAP ) )
	uniform sampler2D specularMap;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshlambert_vert.glsl ////

varying vec3 vViewPosition;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	uniform mat3 uv2Transform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshmatcap_frag.glsl ////

uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshmatcap_vert.glsl ////

varying vec3 vViewPosition;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshnormal_frag.glsl ////

uniform float opacity;
#if ( defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( TANGENTSPACE_NORMALMAP ) )
	varying vec3 vViewPosition;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshnormal_vert.glsl ////

#if ( defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( TANGENTSPACE_NORMALMAP ) )
	varying vec3 vViewPosition;
#endif
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshphong_frag.glsl ////

uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_AOMAP ) )
	uniform sampler2D aoMap;
#endif
#if ( defined( USE_AOMAP ) )
	uniform float aoMapIntensity;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform sampler2D lightMap;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform float lightMapIntensity;
#endif
#if ( defined( USE_EMISSIVEMAP ) )
	uniform sampler2D emissiveMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float envMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float flipEnvMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENVMAP_TYPE_CUBE ) )
	uniform samplerCube envMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENVMAP_TYPE_CUBE ) )
	uniform sampler2D envMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float reflectivity;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
#if ( NUM_DIR_LIGHTS > 0 )
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif
#if ( NUM_POINT_LIGHTS > 0 )
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHTS > 0 )
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_1; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_2; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if ( NUM_HEMI_LIGHTS > 0 )
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
varying vec3 vViewPosition;
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_MAPS > 0 )
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_SPECULARMAP ) )
	uniform sampler2D specularMap;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshphong_vert.glsl ////

varying vec3 vViewPosition;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	uniform mat3 uv2Transform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENV_WORLDPOS ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	varying vec3 vReflect;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENV_WORLDPOS ) )
	uniform float refractionRatio;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshphysical_frag.glsl ////

uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#if ( defined( IOR ) )
	uniform float ior;
#endif
#if ( defined( SPECULAR ) )
	uniform float specularIntensity;
#endif
#if ( defined( SPECULAR ) )
	uniform vec3 specularColor;
#endif
#if ( defined( SPECULAR ) ) && ( defined( USE_SPECULARINTENSITYMAP ) )
	uniform sampler2D specularIntensityMap;
#endif
#if ( defined( SPECULAR ) ) && ( defined( USE_SPECULARCOLORMAP ) )
	uniform sampler2D specularColorMap;
#endif
#if ( defined( USE_CLEARCOAT ) )
	uniform float clearcoat;
#endif
#if ( defined( USE_CLEARCOAT ) )
	uniform float clearcoatRoughness;
#endif
#if ( defined( USE_IRIDESCENCE ) )
	uniform float iridescence;
#endif
#if ( defined( USE_IRIDESCENCE ) )
	uniform float iridescenceIOR;
#endif
#if ( defined( USE_IRIDESCENCE ) )
	uniform float iridescenceThicknessMinimum;
#endif
#if ( defined( USE_IRIDESCENCE ) )
	uniform float iridescenceThicknessMaximum;
#endif
#if ( defined( USE_SHEEN ) )
	uniform vec3 sheenColor;
#endif
#if ( defined( USE_SHEEN ) )
	uniform float sheenRoughness;
#endif
#if ( defined( USE_SHEEN ) ) && ( defined( USE_SHEENCOLORMAP ) )
	uniform sampler2D sheenColorMap;
#endif
#if ( defined( USE_SHEEN ) ) && ( defined( USE_SHEENROUGHNESSMAP ) )
	uniform sampler2D sheenRoughnessMap;
#endif
varying vec3 vViewPosition;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_AOMAP ) )
	uniform sampler2D aoMap;
#endif
#if ( defined( USE_AOMAP ) )
	uniform float aoMapIntensity;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform sampler2D lightMap;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform float lightMapIntensity;
#endif
#if ( defined( USE_EMISSIVEMAP ) )
	uniform sampler2D emissiveMap;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float envMapIntensity;
#endif
#if ( defined( USE_ENVMAP ) )
	uniform float flipEnvMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ( defined( ENVMAP_TYPE_CUBE ) )
	uniform samplerCube envMap;
#endif
#if ( defined( USE_ENVMAP ) ) && ! ( defined( ENVMAP_TYPE_CUBE ) )
	uniform sampler2D envMap;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
#if ( NUM_DIR_LIGHTS > 0 )
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif
#if ( NUM_POINT_LIGHTS > 0 )
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHTS > 0 )
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_1; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_2; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if ( NUM_HEMI_LIGHTS > 0 )
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform float transmission;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform float thickness;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform float attenuationDistance;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform vec3 attenuationColor;
#endif
#if ( defined( USE_TRANSMISSION ) ) && ( defined( USE_TRANSMISSIONMAP ) )
	uniform sampler2D transmissionMap;
#endif
#if ( defined( USE_TRANSMISSION ) ) && ( defined( USE_THICKNESSMAP ) )
	uniform sampler2D thicknessMap;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform vec2 transmissionSamplerSize;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform sampler2D transmissionSamplerMap;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform mat4 modelMatrix;
#endif
#if ( defined( USE_TRANSMISSION ) )
	uniform mat4 projectionMatrix;
#endif
#if ( defined( USE_TRANSMISSION ) )
	varying vec3 vWorldPosition;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_MAPS > 0 )
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_CLEARCOATMAP ) )
	uniform sampler2D clearcoatMap;
#endif
#if ( defined( USE_CLEARCOAT_ROUGHNESSMAP ) )
	uniform sampler2D clearcoatRoughnessMap;
#endif
#if ( defined( USE_CLEARCOAT_NORMALMAP ) )
	uniform sampler2D clearcoatNormalMap;
#endif
#if ( defined( USE_CLEARCOAT_NORMALMAP ) )
	uniform vec2 clearcoatNormalScale;
#endif
#if ( defined( USE_IRIDESCENCEMAP ) )
	uniform sampler2D iridescenceMap;
#endif
#if ( defined( USE_IRIDESCENCE_THICKNESSMAP ) )
	uniform sampler2D iridescenceThicknessMap;
#endif
#if ( defined( USE_ROUGHNESSMAP ) )
	uniform sampler2D roughnessMap;
#endif
#if ( defined( USE_METALNESSMAP ) )
	uniform sampler2D metalnessMap;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshphysical_vert.glsl ////

varying vec3 vViewPosition;
#if ( defined( USE_TRANSMISSION ) )
	varying vec3 vWorldPosition;
#endif
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	uniform mat3 uv2Transform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// meshtoon_frag.glsl ////

uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_AOMAP ) )
	uniform sampler2D aoMap;
#endif
#if ( defined( USE_AOMAP ) )
	uniform float aoMapIntensity;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform sampler2D lightMap;
#endif
#if ( defined( USE_LIGHTMAP ) )
	uniform float lightMapIntensity;
#endif
#if ( defined( USE_EMISSIVEMAP ) )
	uniform sampler2D emissiveMap;
#endif
#if ( defined( USE_GRADIENTMAP ) )
	uniform sampler2D gradientMap;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
#if ( NUM_DIR_LIGHTS > 0 )
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif
#if ( NUM_POINT_LIGHTS > 0 )
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHTS > 0 )
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_1; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_2; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if ( NUM_HEMI_LIGHTS > 0 )
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
varying vec3 vViewPosition;
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_MAPS > 0 )
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform sampler2D bumpMap;
#endif
#if ( defined( USE_BUMPMAP ) )
	uniform float bumpScale;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform sampler2D normalMap;
#endif
#if ( defined( USE_NORMALMAP ) )
	uniform vec2 normalScale;
#endif
#if ( defined( OBJECTSPACE_NORMALMAP ) )
	uniform mat3 normalMatrix;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// meshtoon_vert.glsl ////

varying vec3 vViewPosition;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	varying vec2 vUv2;
#endif
#if ( defined( USE_LIGHTMAP ) || defined( USE_AOMAP ) )
	uniform mat3 uv2Transform;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform sampler2D displacementMap;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementScale;
#endif
#if ( defined( USE_DISPLACEMENTMAP ) )
	uniform float displacementBias;
#endif
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( ! defined( FLAT_SHADED ) )
	varying vec3 vNormal;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vTangent;
#endif
#if ( ! defined( FLAT_SHADED ) ) && ( defined( USE_TANGENT ) )
	varying vec3 vBitangent;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// points_frag.glsl ////

uniform vec3 diffuse;
uniform float opacity;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_MAP ) || defined( USE_ALPHAMAP ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// points_vert.glsl ////

uniform float size;
uniform float scale;
#if ( defined( USE_COLOR_ALPHA ) )
	varying vec4 vColor;
#endif
#if ! ( defined( USE_COLOR_ALPHA ) ) && ( defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) )
	varying vec3 vColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// shadow_frag.glsl ////

uniform vec3 color;
uniform float opacity;
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
uniform bool receiveShadow;
uniform vec3 ambientLightColor;
uniform vec3 lightProbe[ 9 ];
#if ( NUM_DIR_LIGHTS > 0 )
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
#endif
#if ( NUM_POINT_LIGHTS > 0 )
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHTS > 0 )
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_1; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform sampler2D ltc_2; // RGBA Float
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 )
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if ( NUM_HEMI_LIGHTS > 0 )
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_MAPS > 0 )
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif


//// shadow_vert.glsl ////

#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_MORPHTARGETS ) )
	uniform float morphTargetBaseInfluence;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform sampler2DArray morphTargetsTexture;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ( defined( MORPHTARGETS_TEXTURE ) )
	uniform ivec2 morphTargetsTextureSize;
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 8 ];
#endif
#if ( defined( USE_MORPHTARGETS ) ) && ! ( defined( MORPHTARGETS_TEXTURE ) ) && ! ( ! defined( USE_MORPHNORMALS ) )
	uniform float morphTargetInfluences[ 4 ];
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrix;
#endif
#if ( defined( USE_SKINNING ) )
	uniform mat4 bindMatrixInverse;
#endif
#if ( defined( USE_SKINNING ) )
	uniform highp sampler2D boneTexture;
#endif
#if ( defined( USE_SKINNING ) )
	uniform int boneTextureSize;
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( NUM_SPOT_LIGHT_COORDS > 0 )
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
	uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_SPOT_LIGHT_SHADOWS > 0 )
	uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
#endif
#if ( defined( USE_SHADOWMAP ) ) && ( NUM_POINT_LIGHT_SHADOWS > 0 )
	uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
#endif


//// sprite_frag.glsl ////

uniform vec3 diffuse;
uniform float opacity;
#if ( ( defined( USE_UV ) && ! defined( UVS_VERTEX_ONLY ) ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_MAP ) )
	uniform sampler2D map;
#endif
#if ( defined( USE_ALPHAMAP ) )
	uniform sampler2D alphaMap;
#endif
#if ( defined( USE_ALPHATEST ) )
	uniform float alphaTest;
#endif
#if ( defined( USE_FOG ) )
	uniform vec3 fogColor;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_FOG ) ) && ( defined( FOG_EXP2 ) )
	uniform float fogDensity;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogNear;
#endif
#if ( defined( USE_FOG ) ) && ! ( defined( FOG_EXP2 ) )
	uniform float fogFar;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) && defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif


//// sprite_vert.glsl ////

uniform float rotation;
uniform vec2 center;
#if ( defined( USE_UV ) ) && ! ( defined( UVS_VERTEX_ONLY ) )
	varying vec2 vUv;
#endif
#if ( defined( USE_UV ) )
	uniform mat3 uvTransform;
#endif
#if ( defined( USE_FOG ) )
	varying float vFogDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vFragDepth;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ( defined( USE_LOGDEPTHBUF_EXT ) )
	varying float vIsPerspective;
#endif
#if ( defined( USE_LOGDEPTHBUF ) ) && ! ( defined( USE_LOGDEPTHBUF_EXT ) )
	uniform float logDepthBufFC;
#endif
#if ( NUM_CLIPPING_PLANES > 0 )
	varying vec3 vClipPosition;
#endif


//// vsm_frag.glsl ////

uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;


//// vsm_vert.glsl ////




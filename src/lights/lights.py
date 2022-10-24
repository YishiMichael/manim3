from __future__ import annotations

from dataclasses import dataclass

#from colour import Color
import numpy as np

from constants import PI
from lights.ambient_light import AmbientLight
from lights.directional_light import DirectionalLight
from lights.hemisphere_light import HemisphereLight
from lights.hemisphere_light_probe import HemisphereLightProbe
from lights.light import Light
from lights.point_light import PointLight
from lights.rect_area_light import RectAreaLight
from lights.spot_light import SpotLight
from utils.arrays import Mat4, Vec2, Vec3
from utils.spherical_harmonics3 import SphericalHarmonics3
from utils.texture import Texture





#import { Color } from '../../math/Color.js';
#import { Matrix4 } from '../../math/Matrix4.js';
#import { Vector2 } from '../../math/Vector2.js';
#import { Vector3 } from '../../math/Vector3.js';
#import { UniformsLib } from '../shaders/UniformsLib.js';

# function UniformsCache() {

#     const lights = {};

#     return {

#         get: function ( light ) {

#             if ( lights[ light.id ] !== undefined ) {

#                 return lights[ light.id ];

#             }

#             uniforms;

#             switch ( light.type ) {

#                 case 'DirectionalLight':
#                     uniforms = {
#                         direction: new Vector3(),
#                         color: new Color()
#                     };
#                     break;

#                 case 'SpotLight':
#                     uniforms = {
#                         position: new Vector3(),
#                         direction: new Vector3(),
#                         color: new Color(),
#                         distance: 0,
#                         coneCos: 0,
#                         penumbraCos: 0,
#                         decay: 0
#                     };
#                     break;

#                 case 'PointLight':
#                     uniforms = {
#                         position: new Vector3(),
#                         color: new Color(),
#                         distance: 0,
#                         decay: 0
#                     };
#                     break;

#                 case 'HemisphereLight':
#                     uniforms = {
#                         direction: new Vector3(),
#                         skyColor: new Color(),
#                         groundColor: new Color()
#                     };
#                     break;

#                 case 'RectAreaLight':
#                     uniforms = {
#                         color: new Color(),
#                         position: new Vector3(),
#                         halfWidth: new Vector3(),
#                         halfHeight: new Vector3()
#                     };
#                     break;

#             }

#             lights[ light.id ] = uniforms;

#             return uniforms;

#         }

#     };

# }

# function ShadowUniformsCache() {

#     const lights = {};

#     return {

#         get: function ( light ) {

#             if ( lights[ light.id ] !== undefined ) {

#                 return lights[ light.id ];

#             }

#             uniforms;

#             switch ( light.type ) {

#                 case 'DirectionalLight':
#                     uniforms = {
#                         shadowBias: 0,
#                         shadowNormalBias: 0,
#                         shadowRadius: 1,
#                         shadowMapSize: new Vector2()
#                     };
#                     break;

#                 case 'SpotLight':
#                     uniforms = {
#                         shadowBias: 0,
#                         shadowNormalBias: 0,
#                         shadowRadius: 1,
#                         shadowMapSize: new Vector2()
#                     };
#                     break;

#                 case 'PointLight':
#                     uniforms = {
#                         shadowBias: 0,
#                         shadowNormalBias: 0,
#                         shadowRadius: 1,
#                         shadowMapSize: new Vector2(),
#                         shadowCameraNear: 1,
#                         shadowCameraFar: 1000
#                     };
#                     break;

#                 # TODO (abelnation): set RectAreaLight shadow uniforms

#             }

#             lights[ light.id ] = uniforms;

#             return uniforms;

#         }

#     };

# }



# nextVersion = 0;

# function shadowCastingAndTexturingLightsFirst( lightA, lightB ) {

#     return ( lightB.cast_shadow ? 2 : 0 ) - ( lightA.cast_shadow ? 2 : 0 ) + ( lightB.map ? 1 : 0 ) - ( lightA.map ? 1 : 0 );

# }


@dataclass
class DirectionalLightStruct:
    direction: Vec3
    color: Vec3

@dataclass
class DirectionalLightShadowStruct:
    shadowBias: float
    shadowNormalBias: float
    shadowRadius: float
    shadowMapSize: Vec2

@dataclass
class SpotLightStruct:
    position: Vec3
    direction: Vec3
    color: Vec3
    distance: float
    decay: float
    coneCos: float
    penumbraCos: float

@dataclass
class SpotLightShadowStruct:
    shadowBias: float
    shadowNormalBias: float
    shadowRadius: float
    shadowMapSize: Vec2

@dataclass
class RectAreaLightStruct:
    color: Vec3
    position: Vec3
    halfWidth: Vec3
    halfHeight: Vec3

@dataclass
class PointLightStruct:
    position: Vec3
    color: Vec3
    distance: float
    decay: float

@dataclass
class PointLightShadowStruct:
    shadowBias: float
    shadowNormalBias: float
    shadowRadius: float
    shadowMapSize: Vec2
    shadowCameraNear: float
    shadowCameraFar: float

@dataclass
class HemisphereLightStruct:
    direction: Vec3
    skyColor: Vec3
    groundColor: Vec3


@dataclass
class LightsState:
    ambient: Vec3
    probe: SphericalHarmonics3
    directional: list[DirectionalLightStruct]
    directionalShadow: list[DirectionalLightShadowStruct]
    directionalShadowMap: list[Texture | None]
    directionalShadowMatrix: list[Mat4]
    spot: list[SpotLightStruct]
    spotLightMap: list[Texture | None]
    spotShadow: list[SpotLightShadowStruct]
    spotShadowMap: list[Texture | None]
    spotLightMatrix: list[Mat4]
    rectArea: list[RectAreaLightStruct]
    rectAreaLTC1: None  # TODO
    rectAreaLTC2: None  # TODO
    point: list[PointLightStruct]
    pointShadow: list[PointLightShadowStruct]
    pointShadowMap: list[Texture | None]
    pointShadowMatrix: list[Mat4]
    hemi: list[HemisphereLightStruct]
    numSpotLightShadowsWithMaps: int


class Lights:
    def __init__(self):
        self.ambient_lights: list[AmbientLight] = []
        self.hemisphere_light_probes: list[HemisphereLightProbe] = []
        self.directional_lights_cast: list[DirectionalLight] = []
        self.directional_lights_none: list[DirectionalLight] = []
        self.spot_lights_map_cast: list[SpotLight] = []
        self.spot_lights_map_none: list[SpotLight] = []
        self.spot_lights_none_cast: list[SpotLight] = []
        self.spot_lights_none_none: list[SpotLight] = []
        self.rect_area_lights: list[RectAreaLight] = []
        self.point_lights_cast: list[PointLight] = []
        self.point_lights_none: list[PointLight] = []
        self.hemisphere_lights: list[HemisphereLight] = []

    @property
    def directional_lights(self) -> list[DirectionalLight]:
        return self.directional_lights_cast + self.directional_lights_none

    @property
    def spot_lights(self) -> list[SpotLight]:
        return self.spot_lights_map_cast + self.spot_lights_map_none \
            + self.spot_lights_none_cast + self.spot_lights_none_none

    @property
    def spot_lights_map(self) -> list[SpotLight]:
        return self.spot_lights_map_cast + self.spot_lights_map_none

    @property
    def spot_lights_cast(self) -> list[SpotLight]:
        return self.spot_lights_map_cast + self.spot_lights_none_cast

    @property
    def point_lights(self) -> list[PointLight]:
        return self.point_lights_cast + self.point_lights_none

    def add_light(self, light: Light):
        if isinstance(light, AmbientLight):
            self.ambient_lights.append(light)

        elif isinstance(light, HemisphereLightProbe):
            self.hemisphere_light_probes.append(light)

        elif isinstance(light, DirectionalLight):
            if light.cast_shadow:
                self.directional_lights_cast.append(light)
            else:
                self.directional_lights_none.append(light)

        elif isinstance(light, SpotLight):
            if light.map is not None:
                if light.cast_shadow:
                    self.spot_lights_map_cast.append(light)
                else:
                    self.spot_lights_map_none.append(light)
            else:
                if light.cast_shadow:
                    self.spot_lights_none_cast.append(light)
                else:
                    self.spot_lights_none_none.append(light)

        elif isinstance(light, RectAreaLight):
            self.rect_area_lights.append(light)

        elif isinstance(light, PointLight):
            if light.cast_shadow:
                self.point_lights_cast.append(light)
            else:
                self.point_lights_none.append(light)

        elif isinstance(light, HemisphereLight):
            self.hemisphere_lights.append(light)

        return self

    def setup_lights(self, view_matrix: Mat4) -> LightsState:
        ambient: Vec3 = Vec3()
        for light in self.ambient_lights:
            ambient += Vec3(light.color.rgb).scale(light.intensity)

        probe: SphericalHarmonics3 = SphericalHarmonics3()
        for light in self.hemisphere_light_probes:
            color_vec0 = Vec3(light.color.rgb).scale(light.intensity)
            color_vec1 = Vec3(light.ground_color.rgb).scale(light.intensity)
            c0 = np.sqrt(PI) * (color_vec0 + color_vec1)
            c1 = np.sqrt(PI * 0.75) * (color_vec0 - color_vec1)
            probe.coefficients[0] += c0
            probe.coefficients[1] += c1

        directional: list[DirectionalLightStruct] = []
        for light in self.directional_lights:
            directional.append(DirectionalLightStruct(
                direction = Vec3.from_matrix_position(light.matrix - light.target_matrix).transform_direction(view_matrix),
                color = Vec3(light.color.rgb).scale(light.intensity)
            ))
        directionalShadow: list[DirectionalLightShadowStruct] = []
        directionalShadowMap: list[Texture | None] = []
        directionalShadowMatrix: list[Mat4] = []
        for light in self.directional_lights_cast:
            shadow = light.shadow
            directionalShadow.append(DirectionalLightShadowStruct(
                shadowBias = shadow.bias,
                shadowNormalBias = shadow.normalBias,
                shadowRadius = shadow.radius,
                shadowMapSize = shadow.map_size
            ))
            directionalShadowMap.append(shadow.map)
            directionalShadowMatrix.append(shadow.matrix)

        spot: list[SpotLightStruct] = []
        for light in self.spot_lights:
            spot.append(SpotLightStruct(
                position = Vec3.from_matrix_position(light.matrix).apply_affine(view_matrix),
                direction = Vec3.from_matrix_position(light.matrix - light.target_matrix).transform_direction(view_matrix),
                color = Vec3(light.color.rgb).scale(light.intensity),
                distance = light.distance,
                coneCos = np.cos(light.angle),
                penumbraCos = np.cos(light.angle * (1.0 - light.penumbra)),
                decay = light.decay
            ))
        spotLightMap: list[Texture | None] = []
        spotLightMatrix: list[Mat4] = []
        for light in self.spot_lights_map:
            spotLightMap.append(light.map)
            spotLightMatrix.append(light.shadow.matrix)
        spotShadow: list[SpotLightShadowStruct] = []
        spotShadowMap: list[Texture | None] = []
        for light in self.spot_lights_cast:
            shadow = light.shadow
            spotShadow.append(SpotLightShadowStruct(
                shadowBias = shadow.bias,
                shadowNormalBias = shadow.normalBias,
                shadowRadius = shadow.radius,
                shadowMapSize = shadow.map_size
            ))
            spotShadowMap.append(shadow.map)

        rectArea: list[RectAreaLightStruct] = []
        for light in self.rect_area_lights:
            rot_mat = Mat4.from_matrix_rotation(light.matrix.apply(view_matrix))
            rectArea.append(RectAreaLightStruct(
                color = Vec3(light.color.rgb).scale(light.intensity),
                position = Vec3.from_matrix_position(light.matrix).apply_affine(view_matrix),
                halfWidth = Vec3(light.width * 0.5, 0.0, 0.0).apply_affine(rot_mat),
                halfHeight = Vec3(0.0, light.height * 0.5, 0.0).apply_affine(rot_mat)
            ))
        rectAreaLTC1: None = None  # TODO
        rectAreaLTC2: None = None  # TODO

        point: list[PointLightStruct] = []
        for light in self.point_lights:
            point.append(PointLightStruct(
                position = Vec3.from_matrix_position(light.matrix).apply_affine(view_matrix),
                color = Vec3(light.color.rgb).scale(light.intensity),
                distance = light.distance,
                decay = light.decay
            ))
        pointShadow: list[PointLightShadowStruct] = []
        pointShadowMap: list[Texture | None] = []
        pointShadowMatrix: list[Mat4] = []
        for light in self.point_lights_cast:
            shadow = light.shadow
            pointShadow.append(PointLightShadowStruct(
                shadowBias = shadow.bias,
                shadowNormalBias = shadow.normalBias,
                shadowRadius = shadow.radius,
                shadowMapSize = shadow.map_size,
                shadowCameraNear = shadow.camera.near,
                shadowCameraFar = shadow.camera.far
            ))
            pointShadowMap.append(shadow.map)
            pointShadowMatrix.append(shadow.matrix)

        hemi: list[HemisphereLightStruct] = []
        for light in self.hemisphere_lights:
            hemi.append(HemisphereLightStruct(
                direction = Vec3.from_matrix_position(light.matrix).transform_direction(view_matrix),
                skyColor = Vec3(light.color.rgb).scale(light.intensity),
                groundColor = Vec3(light.ground_color.rgb).scale(light.intensity)
            ))

        return LightsState(
            ambient,
            probe,
            directional,
            directionalShadow,
            directionalShadowMap,
            directionalShadowMatrix,
            spot,
            spotLightMap,
            spotShadow,
            spotShadowMap,
            spotLightMatrix,
            rectArea,
            rectAreaLTC1,
            rectAreaLTC2,
            point,
            pointShadow,
            pointShadowMap,
            pointShadowMatrix,
            hemi,
            numSpotLightShadowsWithMaps = len(self.spot_lights_map_cast)
        )

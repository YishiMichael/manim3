from __future__ import annotations
from typing import Any, Callable, Generator, Iterator, TypeVar

from dataclasses import dataclass
import math
import moderngl
import re
from typing import Union

#from scipy.spatial.transform import Rotation

from cameras.camera import Camera
from cameras.orthographic_camera import OrthographicCamera
from constants import *
from fog import Fog, FogExp2, FogLinear
from geometries.geometry import *
from lights.lights import LightsState
from materials.material import *
from mobject import InstancedMesh, Mesh, Mobject, SkinnedMesh, Sprite
# from layers import Layers
from utils.arrays import Mat3, Mat4, Vec2, Vec3, Vec4



T = TypeVar("T")
Self = Any  # TODO: replace in py 3.11


Uniform = Union[
    float,
    int,
    Mat3,
    Mat4,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Texture
]
Attribute = Union[
    list[float],
    list[int],
    list[Mat3],
    list[Mat4],
    list[Vec2],
    list[Vec3],
    list[Vec4]
]


@dataclass
class ShaderData:
    vertex_shader: str
    fragment_shader: str
    uniforms: dict[str, Uniform]
    attributes: dict[str, Attribute]
    render_primitive: int


#def remove_redundancies(lst: Iterable[T], last_occurrance: bool = True) -> list[T]:
#    """
#    Used instead of list(set(l)) to maintain order
#    Keeps the first / last occurrence of each element
#    """
#    lst = list(lst)
#    if last_occurrance:
#        lst = reversed(lst)
#    result = list(dict.fromkeys(lst))
#    if last_occurrance:
#        result = list(reversed(result))
#    return result


# TODO: this should be a tree
class GraphNode:
    """
    A node of a tree

    The algorithm is designed so that every node is unique,
    and loops never exist.
    """
    __slots__ = [
        "parent",
        "children"
    ]

    def __init__(self: Self) -> None:
        self.parent: Self | None = None
        self.children: list[Self] = []

    def __iter__(self: Self) -> Iterator[Self]:
        return iter(self.children)

    #def __getitem__(self: Self, value: int | slice) -> Self:
    #    return self.children.__getitem__(value)

    def get_parent(self: Self) -> Self | None:
        return self.parent

    def get_children(self: Self) -> list[Self]:
        return self.children

    def iter_ancestors(self: Self) -> Generator[Self, None, None]:
        yield self
        if self.parent is not None:
            yield from self.parent.iter_ancestors()

    def iter_descendents(self: Self) -> Generator[Self, None, None]:
        yield self
        for child in self.children:
            yield from child.iter_descendents()

    #def get_ancestors(self: Self) -> list[Self]:  # TODO: order
    #    return remove_redundancies(self.iter_ancestors())

    #def get_descendents(self: Self) -> list[Self]:
    #    return remove_redundancies(self.iter_descendents())

    def includes(self: Self, node: Self) -> bool:
        return self in node.iter_ancestors()

    #@staticmethod
    #def traverse(callback: Callable[[GraphNode], T]) -> Callable[[GraphNode], T]:
    #    def result_func(node: GraphNode) -> T:
    #        result = callback(node)
    #        for child in node.get_descendents():
    #            callback(child)
    #        return result
    #    return result_func

    def _bind_child(self: Self, node: Self, index: int | None = None) -> None:
        if node.includes(self):
            raise ValueError(f"Loop detected when appending '{node}' to '{self}'")
        if self.includes(node):
            node.parent._unbind_child(node)
        if index is None:
            self.children.append(node)
        else:
            self.children.insert(index, node)
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent = self

    def _unbind_child(self: Self, node: Self) -> None:
        self.children.remove(node)
        node.parent = None

    #def clear_bindings(self) -> None:
    #    for parent in self.parent:
    #        parent.children.remove(self)
    #    for child in self.children:
    #        child.parent.remove(self)
    #    #for parent in self.parent:
    #    #    for child in self.children:
    #    #        parent._bind_child(child, loop_check=False)
    #    self.parent.clear()
    #    self.children.clear()

    def index(self: Self, node: Self) -> int:
        return self.children.index(node)

    def insert(self: Self, index: int, *nodes: Self) -> None:
        for i, node in enumerate(nodes, start=index):
            self._bind_child(node, index=i)

    def add(self: Self, *nodes: Self) -> None:
        for node in nodes:
            self._bind_child(node)

    def remove(self: Self, *nodes: Self) -> None:
        for node in nodes:
            self._unbind_child(node)

    def pop(self: Self, index: int = -1) -> Self:
        node = self.children[index]
        self._unbind_child(node)
        return node

    #def clear_parent(self: Self) -> None:
    #    for parent in self.parent:
    #        parent.children.remove(self)
    #    self.parent.clear()

    def clear(self: Self) -> None:
        for child in self.children:
            self._unbind_child(child)

    #def reverse(self: Self) -> None:
    #    self.children.reverse()


#class ShaderConfig:
#    def __init__(self):
#        self.attribute_types: dict[str, type] = {}
#        self.uniform_types: dict[str, type] = {}
#        self.vertex_shader: str = ""
#        self.geometry_shader: str | None = None
#        self.fragment_shader: str = ""
#
#    def 


class Mobject(GraphNode):
    def __init__(self):
        super().__init__()

        #self.parents: list[Mobject] = []
        #self.children: list[Mobject] = []

        #self.position: Vec3 = Vec3()
        #self.quaternion: Rotation = Rotation.identity()
        #self.scalefactor: Vec3 = Vec3()

        self.matrix: Mat4 = Mat4()
        #self._matrix_world: Mat4 = Mat4()
        #self._matrix_world_inverse: Mat4 = Mat4()

        #self.modelViewMatrix: Mat4 = Mat4()
        #self.normalMatrix: Mat3 = Mat3()

        #self.matrixAutoUpdate: bool = True
        #self.matrixWorldNeedsUpdate: bool = False

        #self.matrixWorldAutoUpdate: bool = True

        #self.layers: Layers = Layers()
        #self.visible: bool = True

        #self.cast_shadow: bool = False
        #self.receive_shadow: bool = False

        #self.frustumCulled: bool = True
        #self.render_order: int = 0

        # updaters...?
        self.updaters: list[Callable[[Mobject, float], Mobject]] = []  # TODO

    def set_matrix(self, matrix: Mat4):
        self.matrix = matrix
        #self._matrix_world = (self.parent._matrix_world if self.parent is not None else Mat4()) @ matrix
        #self._matrix_world_inverse = ~self._matrix_world
        return self

    #@property
    #def matrix_world(self) -> Mat4:
    #    result = Mat4()
    #    mob = self
    #    while mob is not None:
    #        result = mob.matrix @ result
    #        mob = mob.parent
    #    return result

    def apply_matrix(self, matrix: Mat4):
        self.set_matrix(self.matrix.apply(matrix))
        return self

    def shift(self, vector: Vec3):
        self.apply_matrix(Mat4.from_shift(vector))
        return self

    def scale(self, scalar: float | Vec3):
        if not isinstance(scalar, Vec3):
            scalar = Vec3(scalar, scalar, scalar)
        self.apply_matrix(Mat4.from_scale(scalar))
        return self

    def setup_shader_data(self, camera: Camera, lights_state: LightsState, fog: Fog) -> ShaderData:
        raise NotImplementedError


#class Mobject(Object3D):
#    
#
#    @staticmethod
#    def get_default_material() -> Material:
#        return Material()  # TODO


class Sprite(Mobject):
    def __init__(self):
        super().__init__()
        self.center: Vec2 = Vec2()


class Mesh(Mobject):
    def __init__(self, geometry: Geometry, material: Material | None = None):
        super().__init__()
        if material is None:
            material = Material()  # TODO

        self.geometry: Geometry = geometry
        self.material: Material = material

    def setup_shader_data(self, camera: Camera, lights_state: LightsState, fog: Fog) -> ShaderData:
        material = self.material
        geometry = self.geometry
        vertex_shader, fragment_shader = acquire_program(
            material = material,
            lights_state = lights_state,
            shadows = [],  # TODO
            fog = fog,
            mobject = self
        )
        uniforms = get_uniforms(
            material = material,
            lights_state = lights_state,
            camera = camera,
            mobject = self,
            fog = fog
        )
        attributes = get_attributes(
            mobject = self,
            geometry = geometry
        )
        return ShaderData(
            vertex_shader = vertex_shader,
            fragment_shader = fragment_shader,
            uniforms = uniforms,
            attributes = attributes,
            render_primitive = moderngl.TRIANGLES
        )


class InstancedMesh(Mesh):
    def __init__(self, geometry: Geometry, material: Material | None = None):
        super().__init__(geometry, material)
        self.instanceMatrix: list = []  # TODO
        self.instanceColor: list | None = None  # TODO


class SkinnedMesh(Mesh):
    pass


#class SkinnedMesh(Mesh):
#    def __init__(self, geometry: Geometry, material: Material | None = None):
#        super().__init__(geometry, material)
#        self.bindMatrix: Mat4 = Mat4()
#        self.bindMatrixInverse: Mat4 = Mat4()



@dataclass
class Define:
    name: str
    active: bool
    value: str | None = None



def read_shader_from_file(filename: str) -> str:
    with open(f"{filename}.glsl") as f:
        result = f.read()
    return result


# Ported from WebGLPrograms.getParameters(), three/src/renderers/webgl/WebGLPrograms
def acquire_program(material: Material, lights_state: LightsState, shadows: list, fog: Fog, mobject: Mobject) -> tuple[str, str]:
    if isinstance(mobject, Mesh):
        geometry = mobject.geometry
    else:
        geometry = None
    #environment = scene.environment if isinstance(material, MeshStandardMaterial) else None

    envMap_ = None  # ( material.isMeshStandardMaterial ? cubeuvmaps : cubemaps ).get( material.envMap || environment )
    envMapCubeUVHeight = None  # ( !! envMap ) && ( envMap.mapping == CubeUVReflectionMapping ) ? envMap.image.height : null

    shaderIDs: dict[type[Material], str] = {
        MeshDepthMaterial: "depth",
        MeshDistanceMaterial: "distanceRGBA",
        MeshNormalMaterial: "meshnormal",  # "normal",
        MeshBasicMaterial: "meshbasic",  # "basic",
        MeshLambertMaterial: "meshlambert",  # "lambert",
        MeshPhongMaterial: "meshphong",  # "phong",
        MeshToonMaterial: "meshtoon",  # "toon",
        MeshStandardMaterial: "meshphysical",  # "physical",
        MeshPhysicalMaterial: "meshphysical",  # "physical",
        MeshMatcapMaterial: "meshmatcap",  # "matcap",
        LineBasicMaterial: "meshbasic",  # "basic",
        LineDashedMaterial: "linedashed",  # "dashed",
        PointsMaterial: "points",
        ShadowMaterial: "shadow",
        SpriteMaterial: "sprite"
    }

    shaderID = shaderIDs.get(type(material))

    # heuristics to create shader parameters according to lights in the scene
    # (not to blow over maxLights budget)

    # if ( material.precision !== null ):

    #     precision = capabilities.getMaxPrecision( material.precision )

    #     if ( precision !== material.precision ):

    #         console.warn( 'THREE.WebGLProgram.getParameters:', material.precision, 'not supported, using', precision, 'instead.' )

    #     }

    # }

    #

    morphTargetsCount = 0 if geometry is None else \
        len(geometry.morphAttributes.position) if geometry.morphAttributes.position else \
        len(geometry.morphAttributes.normal) if geometry.morphAttributes.normal else \
        len(geometry.morphAttributes.color) if geometry.morphAttributes.color else 0

    morphTextureStride = 0
    if geometry is not None:
        if geometry.morphAttributes.position is not None:
            morphTextureStride = 1
        if geometry.morphAttributes.normal is not None:
            morphTextureStride = 2
        if geometry.morphAttributes.color is not None:
            morphTextureStride = 3

    #

    if shaderID is not None:

        # shader = ShaderLib[ shaderID ]

        vertexShader = read_shader_from_file(f"shader_lib/{shaderID}_vert")
        fragmentShader = read_shader_from_file(f"shader_lib/{shaderID}_frag")

    else:
        raise ValueError(f"Cannot handle material of type {type(material)}")

        # vertexShader = material.vertexShader
        # fragmentShader = material.fragmentShader

        # _customShaders.update( material )

        # customVertexShaderID = _customShaders.getVertexShaderID( material )
        # customFragmentShaderID = _customShaders.getFragmentShaderID( material )

    # currentRenderTarget = renderer.getRenderTarget()

    m = convert_to_universal_material(material)

    useAlphaTest = m.alphaTest is not None and m.alphaTest > 0
    useClearcoat = m.clearcoat is not None and m.clearcoat > 0
    useIridescence = m.iridescence is not None and m.iridescence > 0

    #parameters = {

    isWebGL2 = True

    shaderID = shaderID
    shaderName = material.__class__.__name__

    vertexShader = vertexShader
    fragmentShader = fragmentShader

    isRawShaderMaterial = isinstance(material, RawShaderMaterial)
    #glslVersion = GLSL3

    precision = "highp"

    instancing = isinstance(mobject, InstancedMesh)
    instancingColor = isinstance(mobject, InstancedMesh) and mobject.instanceColor is not None

    supportsVertexTextures = True
    outputEncoding = LinearEncoding
    map = m.map is not None
    matcap = m.matcap is not None
    envMap = envMap_ is not None
    envMapMode = envMap_ is not None and envMap_.mapping
    envMapCubeUVHeight = envMapCubeUVHeight
    lightMap = m.lightMap is not None
    aoMap = m.aoMap is not None
    emissiveMap = m.emissiveMap is not None
    bumpMap = m.bumpMap is not None
    normalMap = m.normalMap is not None
    objectSpaceNormalMap = m.normalMapType == ObjectSpaceNormalMap
    tangentSpaceNormalMap = m.normalMapType == TangentSpaceNormalMap

    decodeVideoTexture = False

    clearcoat = useClearcoat
    clearcoatMap = useClearcoat and m.clearcoatMap is not None
    clearcoatRoughnessMap = useClearcoat and m.clearcoatRoughnessMap is not None
    clearcoatNormalMap = useClearcoat and m.clearcoatNormalMap is not None

    iridescence = useIridescence
    iridescenceMap = useIridescence and m.iridescenceMap is not None
    iridescenceThicknessMap = useIridescence and m.iridescenceThicknessMap is not None

    displacementMap = m.displacementMap is not None
    roughnessMap = m.roughnessMap is not None
    metalnessMap = m.metalnessMap is not None
    specularMap = m.specularMap is not None
    specularIntensityMap = m.specularIntensityMap is not None
    specularColorMap = m.specularColorMap is not None

    opaque = not m.transparent and m.blending == NormalBlending

    alphaMap = m.alphaMap is not None
    alphaTest = useAlphaTest

    gradientMap = m.gradientMap is not None

    sheen = m.sheen is not None and m.sheen > 0
    sheenColorMap = m.sheenColorMap is not None
    sheenRoughnessMap = m.sheenRoughnessMap is not None

    transmission = m.transmission is not None and m.transmission > 0
    transmissionMap = m.transmissionMap is not None
    thicknessMap = m.thicknessMap is not None

    combine = m.combine

    vertexTangents = m.normalMap is not None and geometry is not None and geometry.attributes.tangent is not None
    vertexColors = m.vertexColors
    vertexAlphas = m.vertexColors and geometry is not None and geometry.attributes.color and len(geometry.attributes.color[0]) == 4
    vertexUvs = any((
       m.map is not None,
       m.bumpMap is not None,
       m.normalMap is not None,
       m.specularMap is not None,
       m.alphaMap is not None,
       m.emissiveMap is not None,
       m.roughnessMap is not None,
       m.metalnessMap is not None,
       m.clearcoatMap is not None,
       m.clearcoatRoughnessMap is not None,
       m.clearcoatNormalMap is not None,
       m.iridescenceMap is not None,
       m.iridescenceThicknessMap is not None,
       m.displacementMap is not None,
       m.transmissionMap is not None,
       m.thicknessMap is not None,
       m.specularIntensityMap is not None,
       m.specularColorMap is not None,
       m.sheenColorMap is not None,
       m.sheenRoughnessMap is not None
    ))
    uvsVertexOnly = not any((
       m.map is not None,
       m.bumpMap is not None,
       m.normalMap is not None,
       m.specularMap is not None,
       m.alphaMap is not None,
       m.emissiveMap is not None,
       m.roughnessMap is not None,
       m.metalnessMap is not None,
       m.clearcoatNormalMap is not None,
       m.iridescenceMap is not None,
       m.iridescenceThicknessMap is not None,
       m.transmission is not None and m.transmission > 0,
       m.transmissionMap is not None,
       m.thicknessMap is not None,
       m.specularIntensityMap is not None,
       m.specularColorMap is not None,
       m.sheen is not None and m.sheen > 0,
       m.sheenColorMap is not None,
       m.sheenRoughnessMap is not None
    )) and m.displacementMap is not None

    useFog = m.fog is True
    fogExp2 = fog is not None and isinstance(fog, FogExp2)

    flatShading = m.flatShading is not None

    sizeAttenuation = m.sizeAttenuation
    logarithmicDepthBuffer = None

    skinning = isinstance(mobject, SkinnedMesh)

    morphTargets = geometry is not None and geometry.morphAttributes.position is not None
    morphNormals = geometry is not None and geometry.morphAttributes.normal is not None
    morphColors = geometry is not None and geometry.morphAttributes.color is not None
    morphTargetsCount = morphTargetsCount
    morphTextureStride = morphTextureStride

    numDirLights = len(lights_state.directional)
    numPointLights = len(lights_state.point)
    numSpotLights = len(lights_state.spot)
    numSpotLightMaps = len(lights_state.spotLightMap)
    numRectAreaLights = len(lights_state.rectArea)
    numHemiLights = len(lights_state.hemi)

    numDirLightShadows = len(lights_state.directionalShadowMap)
    numPointLightShadows = len(lights_state.pointShadowMap)
    numSpotLightShadows = len(lights_state.spotShadowMap)
    numSpotLightShadowsWithMaps = lights_state.numSpotLightShadowsWithMaps

    numClippingPlanes = 0
    numClipIntersection = 0

    dithering = m.dithering

    shadowMapEnabled = shadows
    shadowMapType = PCFShadowMap

    toneMapping = LinearToneMapping if m.toneMapped else NoToneMapping
    physicallyCorrectLights = True

    premultipliedAlpha = m.premultipliedAlpha

    doubleSided = m.side == DoubleSide
    flipSided = m.side == BackSide

    useDepthPacking = m.depthPacking is not None
    depthPacking = m.depthPacking or 0

    #index0AttributeName = None

    #extensionDerivatives = False
    #extensionFragDepth = False
    #extensionDrawBuffers = False
    #extensionShaderTextureLOD = False

    rendererExtensionFragDepth = True
    #rendererExtensionDrawBuffers = True
    #rendererExtensionShaderTextureLod = True

    #}


    shadowMapTypeDefine = "SHADOWMAP_TYPE_BASIC"
    if shadowMapType == PCFShadowMap:
        shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF"
    elif shadowMapType == PCFSoftShadowMap:
        shadowMapTypeDefine = "SHADOWMAP_TYPE_PCF_SOFT"
    elif shadowMapType == VSMShadowMap:
        shadowMapTypeDefine = "SHADOWMAP_TYPE_VSM"

    envMapTypeDefine = 'ENVMAP_TYPE_CUBE'
    if envMap:
        if envMapMode in (CubeReflectionMapping, CubeRefractionMapping):
            envMapTypeDefine = 'ENVMAP_TYPE_CUBE'
        elif envMapMode == CubeUVReflectionMapping:
            envMapTypeDefine = 'ENVMAP_TYPE_CUBE_UV'

    envMapModeDefine = 'ENVMAP_MODE_REFLECTION'
    if envMap:
        if envMapMode == CubeRefractionMapping:
            envMapModeDefine = 'ENVMAP_MODE_REFRACTION'

    envMapBlendingDefine = 'ENVMAP_BLENDING_NONE';
    if envMap:
        if combine == MultiplyOperation:
            envMapBlendingDefine = 'ENVMAP_BLENDING_MULTIPLY'
        if combine == MixOperation:
            envMapBlendingDefine = 'ENVMAP_BLENDING_MIX'
        if combine == AddOperation:
            envMapBlendingDefine = 'ENVMAP_BLENDING_ADD'

    if envMapCubeUVHeight is not None:
        imageHeight = envMapCubeUVHeight
        maxMip = math.log2(imageHeight) - 2
        texelHeight = 1.0 / imageHeight
        texelWidth = 1.0 / (3 * max(math.pow(2, maxMip), 7 * 16))
    else:
        maxMip = texelHeight = texelWidth = None

    precisionstring = f"precision {precision} float;\nprecision {precision} int;"
    if precision == "highp":
        precisionstring += "\n#define HIGH_PRECISION"
    elif precision == "mediump":
        precisionstring += "\n#define MEDIUM_PRECISION"
    elif precision == "lowp":
        precisionstring += "\n#define LOW_PRECISION"


    if toneMapping == LinearToneMapping:
        toneMappingName = 'Linear'
    elif toneMapping == ReinhardToneMapping:
        toneMappingName = 'Reinhard'
    elif toneMapping == CineonToneMapping:
        toneMappingName = 'OptimizedCineon'
    elif toneMapping == ACESFilmicToneMapping:
        toneMappingName = 'ACESFilmic'
    elif toneMapping == CustomToneMapping:
        toneMappingName = 'Custom'
    else:
        toneMappingName = 'Linear';
    toneMappingFunction = 'vec3 toneMapping( vec3 color ) { return ' + toneMappingName + 'ToneMapping( color ); }'

    if outputEncoding == LinearEncoding:
        component = 'Linear'
    elif outputEncoding ==  sRGBEncoding:
        component = 'sRGB'
    else:
        component = 'Linear'
    texelEncodingFunction = 'vec4 linearToOutputTexel( vec4 value ) { return LinearTo' + component + '( value ); }'


    customExtensions = ""

    #customDefines = "\n".join(
    #    f"#define {name} {value}" if value is not None else f"#define:name}"
    #    for name, value in defines.items()
    #)
    customDefines = ""

    #versionString = f"#version {glslVersion}\n" if glslVersion else ""
    versionString = "#version 330\n\n"

    if isRawShaderMaterial:

        prefixVertex = "\n".join([
            customDefines
        ]) + "\n"

        prefixFragment = "\n".join([
            customExtensions,
            customDefines
        ]) + "\n"

    else:

        prefixVertex = "\n".join([

            precisionstring,

            f'#define SHADER_NAME {shaderName}',

            customDefines,

            "#define USE_INSTANCING" if instancing else "",
            "#define USE_INSTANCING_COLOR" if instancingColor else "",

            "#define VERTEX_TEXTURES" if supportsVertexTextures else "",

            "#define USE_FOG" if useFog and fog is not None else "",
            "#define FOG_EXP2" if useFog and fogExp2 else "",

            "#define USE_MAP" if map else "",
            "#define USE_ENVMAP" if envMap else "",
            "#define " + envMapModeDefine if envMap else "",
            "#define USE_LIGHTMAP" if lightMap else "",
            "#define USE_AOMAP" if aoMap else "",
            "#define USE_EMISSIVEMAP" if emissiveMap else "",
            "#define USE_BUMPMAP" if bumpMap else "",
            "#define USE_NORMALMAP" if normalMap else "",
            "#define OBJECTSPACE_NORMALMAP" if normalMap and objectSpaceNormalMap else "",
            "#define TANGENTSPACE_NORMALMAP" if normalMap and tangentSpaceNormalMap else "",

            "#define USE_CLEARCOATMAP" if clearcoatMap else "",
            "#define USE_CLEARCOAT_ROUGHNESSMAP" if clearcoatRoughnessMap else "",
            "#define USE_CLEARCOAT_NORMALMAP" if clearcoatNormalMap else "",

            "#define USE_IRIDESCENCEMAP" if iridescenceMap else "",
            "#define USE_IRIDESCENCE_THICKNESSMAP" if iridescenceThicknessMap else "",

            "#define USE_DISPLACEMENTMAP" if displacementMap and supportsVertexTextures else "",

            "#define USE_SPECULARMAP" if specularMap else "",
            "#define USE_SPECULARINTENSITYMAP" if specularIntensityMap else "",
            "#define USE_SPECULARCOLORMAP" if specularColorMap else "",

            "#define USE_ROUGHNESSMAP" if roughnessMap else "",
            "#define USE_METALNESSMAP" if metalnessMap else "",
            "#define USE_ALPHAMAP" if alphaMap else "",

            "#define USE_TRANSMISSION" if transmission else "",
            "#define USE_TRANSMISSIONMAP" if transmissionMap else "",
            "#define USE_THICKNESSMAP" if thicknessMap else "",

            "#define USE_SHEENCOLORMAP" if sheenColorMap else "",
            "#define USE_SHEENROUGHNESSMAP" if sheenRoughnessMap else "",

            "#define USE_TANGENT" if vertexTangents else "",
            "#define USE_COLOR" if vertexColors else "",
            "#define USE_COLOR_ALPHA" if vertexAlphas else "",
            "#define USE_UV" if vertexUvs else "",
            "#define UVS_VERTEX_ONLY" if uvsVertexOnly else "",

            "#define FLAT_SHADED" if flatShading else "",

            "#define USE_SKINNING" if skinning else "",

            "#define USE_MORPHTARGETS" if morphTargets else "",
            "#define USE_MORPHNORMALS" if morphNormals and not flatShading else "",
            "#define USE_MORPHCOLORS" if morphColors and isWebGL2 else "",
            "#define MORPHTARGETS_TEXTURE" if morphTargetsCount > 0 and isWebGL2 else "",
            "#define MORPHTARGETS_TEXTURE_STRIDE " + str(morphTextureStride) if morphTargetsCount > 0 and isWebGL2 else "",
            "#define MORPHTARGETS_COUNT " + str(morphTargetsCount) if morphTargetsCount > 0 and isWebGL2 else "",
            "#define DOUBLE_SIDED" if doubleSided else "",
            "#define FLIP_SIDED" if flipSided else "",

            "#define USE_SHADOWMAP" if shadowMapEnabled else "",
            "#define " + shadowMapTypeDefine if shadowMapEnabled else "",

            "#define USE_SIZEATTENUATION" if sizeAttenuation else "",

            "#define USE_LOGDEPTHBUF" if logarithmicDepthBuffer else "",
            "#define USE_LOGDEPTHBUF_EXT" if logarithmicDepthBuffer and rendererExtensionFragDepth else "",

            'uniform mat4 modelMatrix;',
            'uniform mat4 modelViewMatrix;',
            'uniform mat4 projectionMatrix;',
            'uniform mat4 viewMatrix;',
            'uniform mat3 normalMatrix;',
            'uniform vec3 cameraPosition;',
            'uniform bool isOrthographic;',

            '#ifdef USE_INSTANCING',

            '   attribute mat4 instanceMatrix;',

            '#endif',

            '#ifdef USE_INSTANCING_COLOR',

            '   attribute vec3 instanceColor;',

            '#endif',

            'attribute vec3 position;',
            'attribute vec3 normal;',
            'attribute vec2 uv;',

            '#ifdef USE_TANGENT',

            '   attribute vec4 tangent;',

            '#endif',

            '#if defined( USE_COLOR_ALPHA )',

            '   attribute vec4 color;',

            '#elif defined( USE_COLOR )',

            '   attribute vec3 color;',

            '#endif',

            '#if ( defined( USE_MORPHTARGETS ) && ! defined( MORPHTARGETS_TEXTURE ) )',

            '   attribute vec3 morphTarget0;',
            '   attribute vec3 morphTarget1;',
            '   attribute vec3 morphTarget2;',
            '   attribute vec3 morphTarget3;',

            '   #ifdef USE_MORPHNORMALS',

            '       attribute vec3 morphNormal0;',
            '       attribute vec3 morphNormal1;',
            '       attribute vec3 morphNormal2;',
            '       attribute vec3 morphNormal3;',

            '   #else',

            '       attribute vec3 morphTarget4;',
            '       attribute vec3 morphTarget5;',
            '       attribute vec3 morphTarget6;',
            '       attribute vec3 morphTarget7;',

            '   #endif',

            '#endif',

            '#ifdef USE_SKINNING',

            '   attribute vec4 skinIndex;',
            '   attribute vec4 skinWeight;',

            '#endif',

        ]) + "\n"

        prefixFragment = "\n".join([

            customExtensions,

            precisionstring,

            '#define SHADER_NAME ' + shaderName,

            customDefines,

            '#define USE_FOG' if useFog and fog is not None else "",
            '#define FOG_EXP2' if useFog and fogExp2 else "",

            '#define USE_MAP' if map else "",
            '#define USE_MATCAP' if matcap else "",
            '#define USE_ENVMAP' if envMap else "",
            '#define ' + envMapTypeDefine if envMap else "",
            '#define ' + envMapModeDefine if envMap else "",
            '#define ' + envMapBlendingDefine if envMap else "",
            '#define CUBEUV_TEXEL_WIDTH ' + str(texelWidth) if texelWidth is not None else "",
            '#define CUBEUV_TEXEL_HEIGHT ' + str(texelHeight) if texelHeight is not None else "",
            '#define CUBEUV_MAX_MIP ' + str(maxMip) if maxMip is not None else "",
            '#define USE_LIGHTMAP' if lightMap else "",
            '#define USE_AOMAP' if aoMap else "",
            '#define USE_EMISSIVEMAP' if emissiveMap else "",
            '#define USE_BUMPMAP' if bumpMap else "",
            '#define USE_NORMALMAP' if normalMap else "",
            '#define OBJECTSPACE_NORMALMAP' if normalMap and objectSpaceNormalMap else "",
            '#define TANGENTSPACE_NORMALMAP' if normalMap and tangentSpaceNormalMap else "",

            '#define USE_CLEARCOAT' if clearcoat else "",
            '#define USE_CLEARCOATMAP' if clearcoatMap else "",
            '#define USE_CLEARCOAT_ROUGHNESSMAP' if clearcoatRoughnessMap else "",
            '#define USE_CLEARCOAT_NORMALMAP' if clearcoatNormalMap else "",

            '#define USE_IRIDESCENCE' if iridescence else "",
            '#define USE_IRIDESCENCEMAP' if iridescenceMap else "",
            '#define USE_IRIDESCENCE_THICKNESSMAP' if iridescenceThicknessMap else "",

            '#define USE_SPECULARMAP' if specularMap else "",
            '#define USE_SPECULARINTENSITYMAP' if specularIntensityMap else "",
            '#define USE_SPECULARCOLORMAP' if specularColorMap else "",
            '#define USE_ROUGHNESSMAP' if roughnessMap else "",
            '#define USE_METALNESSMAP' if metalnessMap else "",

            '#define USE_ALPHAMAP' if alphaMap else "",
            '#define USE_ALPHATEST' if alphaTest else "",

            '#define USE_SHEEN' if sheen else "",
            '#define USE_SHEENCOLORMAP' if sheenColorMap else "",
            '#define USE_SHEENROUGHNESSMAP' if sheenRoughnessMap else "",

            '#define USE_TRANSMISSION' if transmission else "",
            '#define USE_TRANSMISSIONMAP' if transmissionMap else "",
            '#define USE_THICKNESSMAP' if thicknessMap else "",

            '#define DECODE_VIDEO_TEXTURE' if decodeVideoTexture else "",

            '#define USE_TANGENT' if vertexTangents else "",
            '#define USE_COLOR' if vertexColors or instancingColor else "",
            '#define USE_COLOR_ALPHA' if vertexAlphas else "",
            '#define USE_UV' if vertexUvs else "",
            '#define UVS_VERTEX_ONLY' if uvsVertexOnly else "",

            '#define USE_GRADIENTMAP' if gradientMap else "",

            '#define FLAT_SHADED' if flatShading else "",

            '#define DOUBLE_SIDED' if doubleSided else "",
            '#define FLIP_SIDED' if flipSided else "",

            '#define USE_SHADOWMAP' if shadowMapEnabled else "",
            '#define ' + shadowMapTypeDefine if shadowMapEnabled else "",

            '#define PREMULTIPLIED_ALPHA' if premultipliedAlpha else "",

            '#define PHYSICALLY_CORRECT_LIGHTS' if physicallyCorrectLights else "",

            '#define USE_LOGDEPTHBUF' if logarithmicDepthBuffer else "",
            '#define USE_LOGDEPTHBUF_EXT' if logarithmicDepthBuffer and rendererExtensionFragDepth else "",

            'uniform mat4 viewMatrix;',
            'uniform vec3 cameraPosition;',
            'uniform bool isOrthographic;',

            '#define TONE_MAPPING' if toneMapping != NoToneMapping else '',
            read_shader_from_file("shader_chunk/tonemapping_pars_fragment") if toneMapping != NoToneMapping else '', # this code is required here because it is used by the toneMapping() function defined below
            toneMappingFunction if toneMapping != NoToneMapping else '',

            '#define DITHERING' if dithering else '',
            '#define OPAQUE' if opaque else '',

            read_shader_from_file("shader_chunk/encodings_pars_fragment"), # this code is required here because it is used by the various encoding/decoding function defined below
            texelEncodingFunction,

            '#define DEPTH_PACKING ' + str(depthPacking) if useDepthPacking else '',

        ]) + "\n"

    def replace_macros_by_nums(macro_map: dict[str, int], s: str) -> str:
        for macro_name, num in macro_map.items():
            s = re.sub(f"\\b{macro_name}\\b", str(num), s)
        return s

    def resolveIncludes(s: str) -> str:
        return re.sub(
            r"^\s*#include\s*<([\w./]+?)>",
            lambda m: read_shader_from_file(f"shader_chunk/{m.group(1)}"),
            s,
            flags=re.MULTILINE
        )

    def replaceLightNums(s: str) -> str:
        return replace_macros_by_nums({
            "NUM_DIR_LIGHTS": numDirLights,
            "NUM_SPOT_LIGHTS": numSpotLights,
            "NUM_SPOT_LIGHT_MAPS": numSpotLightMaps,
            "NUM_SPOT_LIGHT_COORDS": numSpotLightShadows + numSpotLightMaps - numSpotLightShadowsWithMaps,
            "NUM_RECT_AREA_LIGHTS": numRectAreaLights,
            "NUM_POINT_LIGHTS": numPointLights,
            "NUM_HEMI_LIGHTS": numHemiLights,
            "NUM_DIR_LIGHT_SHADOWS": numDirLightShadows,
            "NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS": numSpotLightShadowsWithMaps,
            "NUM_SPOT_LIGHT_SHADOWS": numSpotLightShadows,
            "NUM_POINT_LIGHT_SHADOWS": numPointLightShadows
        }, s)

    def replaceClippingPlaneNums(s: str) -> str:
        return replace_macros_by_nums({
            "NUM_CLIPPING_PLANES": numClippingPlanes,
            "UNION_CLIPPING_PLANES": numClippingPlanes - numClipIntersection
        }, s)

    def unrollLoops(s: str) -> str:
        return re.sub(
            r"#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end",
            lambda m: "".join(
                replace_macros_by_nums(
                    {"UNROLLED_LOOP_INDEX": i},
                    re.sub(r"\[\s*i\s*\]", f"[ {i} ]", m.group(3))
                )
                for i in range(int(m.group(1)), int(m.group(2)))
            ),
            s,
            flags=re.MULTILINE
        )

    def filter_empty_lines(s: str) -> str:
        return "\n".join(
            filter(lambda line: line.strip(), s.split("\n"))
        ) + "\n"

    vertexShader = resolveIncludes(vertexShader)
    vertexShader = replaceLightNums(vertexShader)
    vertexShader = replaceClippingPlaneNums(vertexShader)
    vertexShader = unrollLoops(vertexShader)

    fragmentShader = resolveIncludes(fragmentShader)
    fragmentShader = replaceLightNums(fragmentShader)
    fragmentShader = replaceClippingPlaneNums(fragmentShader)
    fragmentShader = unrollLoops(fragmentShader)


    if isWebGL2 and not isRawShaderMaterial:

        # GLSL 3.0 conversion for built-in materials and ShaderMaterial

        versionString = '#version 300 es\n';

        prefixVertex = "\n".join([
            'precision mediump sampler2DArray;',
            '#define attribute in',
            '#define varying out',
            '#define texture2D texture',
            prefixVertex
        ])

        prefixFragment = "\n".join([
            '#define varying in',
            'layout(location = 0) out highp vec4 pc_fragColor;',  # Assumed glslVersion != GLSL3
            '#define gl_FragColor pc_fragColor',  # Assumed glslVersion != GLSL3
            '#define gl_FragDepthEXT gl_FragDepth',
            '#define texture2D texture',
            '#define textureCube texture',
            '#define texture2DProj textureProj',
            '#define texture2DLodEXT textureLod',
            '#define texture2DProjLodEXT textureProjLod',
            '#define textureCubeLodEXT textureLod',
            '#define texture2DGradEXT textureGrad',
            '#define texture2DProjGradEXT textureProjGrad',
            '#define textureCubeGradEXT textureGrad',
            prefixFragment
        ])


    vertexGlsl = filter_empty_lines(versionString + prefixVertex + vertexShader)
    fragmentGlsl = filter_empty_lines(versionString + prefixFragment + fragmentShader)
    return vertexGlsl, fragmentGlsl


def refreshMaterialUniforms(uniforms: dict, material: Material, pixelRatio: float, height: float, transmissionRenderTarget: None):

    def refreshUniformsCommon( uniforms, material ):

        uniforms["opacity"] = material.opacity

        if ( material.color ):

            uniforms["diffuse"] = ( material.color )

        if ( material.emissive ):

            uniforms["emissive"] = ( material.emissive ).multiplyScalar( material.emissiveIntensity )

        if ( material.map ):

            uniforms["map"] = material.map

        if ( material.alphaMap ):

            uniforms["alphaMap"] = material.alphaMap

        if ( material.bumpMap ):

            uniforms["bumpMap"] = material.bumpMap
            uniforms["bumpScale"] = material.bumpScale
            if ( material.side == BackSide ):
                uniforms["bumpScale"] *= - 1

        if ( material.displacementMap ):

            uniforms["displacementMap"] = material.displacementMap
            uniforms["displacementScale"] = material.displacementScale
            uniforms["displacementBias"] = material.displacementBias

        if ( material.emissiveMap ):

            uniforms["emissiveMap"] = material.emissiveMap

        if ( material.normalMap ):

            uniforms["normalMap"] = material.normalMap
            uniforms["normalScale"] = ( material.normalScale )
            if ( material.side == BackSide ):
                uniforms["normalScale"].negate()

        if ( material.specularMap ):

            uniforms["specularMap"] = material.specularMap

        if ( material.alphaTest > 0 ):

            uniforms["alphaTest"] = material.alphaTest

        #envMap = None

        #if ( envMap ):

        #    uniforms["envMap"] = envMap

        #    uniforms["flipEnvMap"] = -1 if (isinstance(envMap, CubeTexture) and not envMap.isRenderTargetTexture) else 1

        #    uniforms["reflectivity"] = material.reflectivity
        #    uniforms["ior"] = material.ior
        #    uniforms["refractionRatio"] = material.refractionRatio

        if ( material.lightMap ):

            uniforms["lightMap"] = material.lightMap

            # artist-friendly light intensity scaling factor
            physicallyCorrectLights = True
            scaleFactor = 1.0 if physicallyCorrectLights else math.pi

            uniforms["lightMapIntensity"] = material.lightMapIntensity * scaleFactor

        if ( material.aoMap ):

            uniforms["aoMap"] = material.aoMap
            uniforms["aoMapIntensity"] = material.aoMapIntensity

        # uv repeat and offset setting priorities
        # 1. color map
        # 2. specular map
        # 3. displacementMap map
        # 4. normal map
        # 5. bump map
        # 6. roughnessMap map
        # 7. metalnessMap map
        # 8. alphaMap map
        # 9. emissiveMap map
        # 10. clearcoat map
        # 11. clearcoat normal map
        # 12. clearcoat roughnessMap map
        # 13. iridescence map
        # 14. iridescence thickness map
        # 15. specular intensity map
        # 16. specular tint map
        # 17. transmission map
        # 18. thickness map


        if ( material.map ):

            uvScaleMap = material.map

        elif ( material.specularMap ):

            uvScaleMap = material.specularMap

        elif ( material.displacementMap ):

            uvScaleMap = material.displacementMap

        elif ( material.normalMap ):

            uvScaleMap = material.normalMap

        elif ( material.bumpMap ):

            uvScaleMap = material.bumpMap

        elif ( material.roughnessMap ):

            uvScaleMap = material.roughnessMap

        elif ( material.metalnessMap ):

            uvScaleMap = material.metalnessMap

        elif ( material.alphaMap ):

            uvScaleMap = material.alphaMap

        elif ( material.emissiveMap ):

            uvScaleMap = material.emissiveMap

        elif ( material.clearcoatMap ):

            uvScaleMap = material.clearcoatMap

        elif ( material.clearcoatNormalMap ):

            uvScaleMap = material.clearcoatNormalMap

        elif ( material.clearcoatRoughnessMap ):

            uvScaleMap = material.clearcoatRoughnessMap

        elif ( material.iridescenceMap ):

            uvScaleMap = material.iridescenceMap

        elif ( material.iridescenceThicknessMap ):

            uvScaleMap = material.iridescenceThicknessMap

        elif ( material.specularIntensityMap ):

            uvScaleMap = material.specularIntensityMap

        elif ( material.specularColorMap ):

            uvScaleMap = material.specularColorMap

        elif ( material.transmissionMap ):

            uvScaleMap = material.transmissionMap

        elif ( material.thicknessMap ):

            uvScaleMap = material.thicknessMap

        elif ( material.sheenColorMap ):

            uvScaleMap = material.sheenColorMap

        elif ( material.sheenRoughnessMap ):

            uvScaleMap = material.sheenRoughnessMap

        else:
            uvScaleMap = None

        if uvScaleMap is not None:
            # backwards compatibility
            #if ( uvScaleMap.isWebGLRenderTarget ):

            #    uvScaleMap = uvScaleMap.texture

            #if ( uvScaleMap.matrixAutoUpdate ):

            #    uvScaleMap.updateMatrix()

            uniforms["uvTransform"] = ( uvScaleMap.matrix )

        # uv repeat and offset setting priorities for uv2
        # 1. ao map
        # 2. light map

        if ( material.aoMap ):

            uv2ScaleMap = material.aoMap

        elif ( material.lightMap ):

            uv2ScaleMap = material.lightMap

        else:
            uv2ScaleMap = None

        if uv2ScaleMap is not None:
            # backwards compatibility
            #if ( uv2ScaleMap.isWebGLRenderTarget ):

            #    uv2ScaleMap = uv2ScaleMap.texture


            #if ( uv2ScaleMap.matrixAutoUpdate ):

            #    uv2ScaleMap.updateMatrix()

            uniforms["uv2Transform"] = ( uv2ScaleMap.matrix )


    def refreshUniformsLine( uniforms, material ):

        uniforms["diffuse"] = ( material.color )
        uniforms["opacity"] = material.opacity


    def refreshUniformsDash( uniforms, material ):

        uniforms["dashSize"] = material.dashSize
        uniforms["totalSize"] = material.dashSize + material.gapSize
        uniforms["scale"] = material.scale


    def refreshUniformsPoints( uniforms, material, pixelRatio, height ):

        uniforms["diffuse"] = ( material.color )
        uniforms["opacity"] = material.opacity
        uniforms["size"] = material.size * pixelRatio
        uniforms["scale"] = height * 0.5

        if ( material.map ):

            uniforms["map"] = material.map

        if ( material.alphaMap ):

            uniforms["alphaMap"] = material.alphaMap

        if ( material.alphaTest > 0 ):

            uniforms["alphaTest"] = material.alphaTest

        # uv repeat and offset setting priorities
        # 1. color map
        # 2. alpha map

        if ( material.map ):

            uvScaleMap = material.map

        elif ( material.alphaMap ):

            uvScaleMap = material.alphaMap

        else:
            uvScaleMap = None

        if uvScaleMap is not None:
            #if ( uvScaleMap.matrixAutoUpdate == true ):

            #    uvScaleMap.updateMatrix()

            uniforms["uvTransform"] = ( uvScaleMap.matrix )


    def refreshUniformsSprites( uniforms, material ):

        uniforms["diffuse"] = ( material.color )
        uniforms["opacity"] = material.opacity
        uniforms["rotation"] = material.rotation

        if ( material.map ):

            uniforms["map"] = material.map

        if ( material.alphaMap ):

            uniforms["alphaMap"] = material.alphaMap

        if ( material.alphaTest > 0 ):

            uniforms["alphaTest"] = material.alphaTest

        # uv repeat and offset setting priorities
        # 1. color map
        # 2. alpha map

        if ( material.map ):

            uvScaleMap = material.map

        elif ( material.alphaMap ):

            uvScaleMap = material.alphaMap

        else:
            uvScaleMap = None

        if uvScaleMap is not None:
            #if ( uvScaleMap.matrixAutoUpdate == true ):

            #    uvScaleMap.updateMatrix()

            uniforms["uvTransform"] = ( uvScaleMap.matrix )


    def refreshUniformsPhong( uniforms, material ):

        uniforms["specular"] = ( material.specular )
        uniforms["shininess"] = max( material.shininess, 1e-4 ) # to prevent pow( 0.0, 0.0 )


    def refreshUniformsToon( uniforms, material ):

        if ( material.gradientMap ):

            uniforms["gradientMap"] = material.gradientMap


    def refreshUniformsStandard( uniforms, material ):

        uniforms["roughness"] = material.roughness
        uniforms["metalness"] = material.metalness

        if ( material.roughnessMap ):

            uniforms["roughnessMap"] = material.roughnessMap

        if ( material.metalnessMap ):

            uniforms["metalnessMap"] = material.metalnessMap

        envMap = None

        if ( envMap ):

            #uniforms["envMap"] = material.envMap # part of uniforms common
            uniforms["envMapIntensity"] = material.envMapIntensity


    def refreshUniformsPhysical( uniforms, material, transmissionRenderTarget ):

        uniforms["ior"] = material.ior # also part of uniforms common

        if ( material.sheen > 0 ):

            uniforms["sheenColor"] = ( material.sheenColor ).multiplyScalar( material.sheen )

            uniforms["sheenRoughness"] = material.sheenRoughness

            if ( material.sheenColorMap ):

                uniforms["sheenColorMap"] = material.sheenColorMap

            if ( material.sheenRoughnessMap ):

                uniforms["sheenRoughnessMap"] = material.sheenRoughnessMap


        if ( material.clearcoat > 0 ):

            uniforms["clearcoat"] = material.clearcoat
            uniforms["clearcoatRoughness"] = material.clearcoatRoughness

            if ( material.clearcoatMap ):

                uniforms["clearcoatMap"] = material.clearcoatMap

            if ( material.clearcoatRoughnessMap ):

                uniforms["clearcoatRoughnessMap"] = material.clearcoatRoughnessMap

            if ( material.clearcoatNormalMap ):

                uniforms["clearcoatNormalScale"] = ( material.clearcoatNormalScale )
                uniforms["clearcoatNormalMap"] = material.clearcoatNormalMap

                if ( material.side == BackSide ):

                    uniforms["clearcoatNormalScale"].negate()



        if ( material.iridescence > 0 ):

            uniforms["iridescence"] = material.iridescence
            uniforms["iridescenceIOR"] = material.iridescenceIOR
            uniforms["iridescenceThicknessMinimum"] = material.iridescenceThicknessRange[ 0 ]
            uniforms["iridescenceThicknessMaximum"] = material.iridescenceThicknessRange[ 1 ]

            if ( material.iridescenceMap ):

                uniforms["iridescenceMap"] = material.iridescenceMap

            if ( material.iridescenceThicknessMap ):

                uniforms["iridescenceThicknessMap"] = material.iridescenceThicknessMap

        if ( material.transmission > 0 ):

            uniforms["transmission"] = material.transmission
            uniforms["transmissionSamplerMap"] = transmissionRenderTarget.texture
            uniforms["transmissionSamplerSize"].set( transmissionRenderTarget.width, transmissionRenderTarget.height )

            if ( material.transmissionMap ):

                uniforms["transmissionMap"] = material.transmissionMap

            uniforms["thickness"] = material.thickness

            if ( material.thicknessMap ):

                uniforms["thicknessMap"] = material.thicknessMap

            uniforms["attenuationDistance"] = material.attenuationDistance
            uniforms["attenuationColor"] = ( material.attenuationColor )

        uniforms["specularIntensity"] = material.specularIntensity
        uniforms["specularColor"] = ( material.specularColor )

        if ( material.specularIntensityMap ):

            uniforms["specularIntensityMap"] = material.specularIntensityMap

        if ( material.specularColorMap ):

            uniforms["specularColorMap"] = material.specularColorMap

    def refreshUniformsMatcap( uniforms, material ):

        if ( material.matcap ):

            uniforms["matcap"] = material.matcap

    def refreshUniformsDistance( uniforms, material ):

        uniforms["referencePosition"] = ( material.referencePosition )
        uniforms["nearDistance"] = material.nearDistance
        uniforms["farDistance"] = material.farDistance

    m = convert_to_universal_material(material)
    if isinstance(material, MeshBasicMaterial):
        refreshUniformsCommon( uniforms, m )
    elif isinstance(material, MeshLambertMaterial):
        refreshUniformsCommon( uniforms, m )
    elif isinstance(material, MeshToonMaterial):
        refreshUniformsCommon( uniforms, m )
        refreshUniformsToon( uniforms, m )
    elif isinstance(material, MeshPhongMaterial):
        refreshUniformsCommon( uniforms, m )
        refreshUniformsPhong( uniforms, m )
    elif isinstance(material, MeshStandardMaterial):
        refreshUniformsCommon( uniforms, m )
        refreshUniformsStandard( uniforms, m )
        if isinstance(material, MeshPhysicalMaterial):
            refreshUniformsPhysical( uniforms, m, transmissionRenderTarget )
    elif isinstance(material, MeshMatcapMaterial):
        refreshUniformsCommon( uniforms, m )
        refreshUniformsMatcap( uniforms, m )
    elif isinstance(material, MeshDepthMaterial):
        refreshUniformsCommon( uniforms, m )
    elif isinstance(material, MeshDistanceMaterial):
        refreshUniformsCommon( uniforms, m )
        refreshUniformsDistance( uniforms, m )
    elif isinstance(material, MeshNormalMaterial):
        refreshUniformsCommon( uniforms, m )
    elif isinstance(material, LineBasicMaterial):
        refreshUniformsLine( uniforms, m )
        if isinstance(material, LineDashedMaterial):
            refreshUniformsDash( uniforms, m )
    elif isinstance(material, PointsMaterial):
        refreshUniformsPoints( uniforms, m, pixelRatio, height )
    elif isinstance(material, SpriteMaterial):
        refreshUniformsSprites( uniforms, m )
    elif isinstance(material, ShadowMaterial):
        uniforms["color"] = ( m.color )
        uniforms["opacity"] = m.opacity
    #elif ( material.isShaderMaterial ):

    #    material.uniformsNeedUpdate = false # #15581



def get_uniforms(material: Material, lights_state: LightsState, camera: Camera, mobject: Mobject, fog: Fog) -> dict:
    shaderIDs: dict[type[Material], type[Uniforms]] = {
        MeshDepthMaterial: DepthUniform,
        MeshDistanceMaterial: DistanceRGBAUniform,
        MeshNormalMaterial: NormalUniform,
        MeshBasicMaterial: BasicUniform,
        MeshLambertMaterial: LambertUniform,
        MeshPhongMaterial: PhongUniform,
        MeshToonMaterial: ToonUniform,
        MeshStandardMaterial: StandardUniform,
        MeshPhysicalMaterial: PhysicalUniform,
        MeshMatcapMaterial: MatcapUniform,
        LineBasicMaterial: BasicUniform,
        LineDashedMaterial: DashedUniform,
        PointsMaterial: PointsUniform,
        ShadowMaterial: ShadowUniform,
        SpriteMaterial: SpriteUniform
    }

    uniform_class = shaderIDs.get(type(material))
    if uniform_class is None:
        raise ValueError(f"Cannot handle material of type {type(material)}")

    uniforms = {
        k: v
        for k, v in uniform_class().__dict__.items()
        if not k.startswith("__")
    }


    #if not isinstance(material, (ShaderMaterial, RawShaderMaterial)) or material.clipping:
    #    uniforms.clippingPlanes = clipping.uniform;

    #updateCommonMaterialProperties( material, parameters );

    # store the light setup it was created for

    #materialProperties.needsLights = materialNeedsLights( material );
    #materialProperties.lightsStateVersion = lightsStateVersion;

    if any((
        isinstance(material, MeshLambertMaterial),
        isinstance(material, MeshToonMaterial),
        isinstance(material, MeshPhongMaterial),
        isinstance(material, MeshStandardMaterial),
        isinstance(material, ShadowMaterial)
        #isinstance(material, ShaderMaterial) and material.lights
    )):

        # wire up the material to this renderer's lighting state

        uniforms["ambientLightColor"] = lights_state.ambient
        uniforms["lightProbe"] = lights_state.probe
        uniforms["directionalLights"] = lights_state.directional
        uniforms["directionalLightShadows"] = lights_state.directionalShadow
        uniforms["spotLights"] = lights_state.spot
        uniforms["spotLightShadows"] = lights_state.spotShadow
        uniforms["rectAreaLights"] = lights_state.rectArea
        uniforms["ltc_1"] = lights_state.rectAreaLTC1
        uniforms["ltc_2"] = lights_state.rectAreaLTC2
        uniforms["pointLights"] = lights_state.point
        uniforms["pointLightShadows"] = lights_state.pointShadow
        uniforms["hemisphereLights"] = lights_state.hemi

        uniforms["directionalShadowMap"] = lights_state.directionalShadowMap
        uniforms["directionalShadowMatrix"] = lights_state.directionalShadowMatrix
        uniforms["spotShadowMap"] = lights_state.spotShadowMap
        uniforms["spotLightMatrix"] = lights_state.spotLightMatrix
        uniforms["spotLightMap"] = lights_state.spotLightMap
        uniforms["pointShadowMap"] = lights_state.pointShadowMap
        uniforms["pointShadowMatrix"] = lights_state.pointShadowMatrix
        # TODO (abelnation): add area lights shadow info to uniforms




    #const p_uniforms = program.get_uniforms(),
    #    m_uniforms = materialProperties.uniforms;



    uniforms['projectionMatrix'] = camera.projection_matrix

    uniforms['logDepthBufFC'] = 2.0 / ( math.log( camera.far + 1.0 ) / math.log(2) )



    # load material specific uniforms
    # (shader material also gets them for the sake of genericity)

    #if isinstance(material, (MeshPhongMaterial, MeshToonMaterial, MeshStandardMaterial)):

    #    uCamPos = uniforms["map"].cameraPosition;

    #    if ( uCamPos !== undefined ):

    #        uCamPos.setValue( _gl,
    #            _vector3.setFromMatrixPosition( camera.matrixWorld ) );

    #    }

    #}

    if isinstance(material, (
        MeshPhongMaterial,
        MeshToonMaterial,
        MeshLambertMaterial,
        MeshBasicMaterial,
        MeshStandardMaterial,
        #ShaderMaterial
    )):

        uniforms['isOrthographic'] = isinstance(camera, OrthographicCamera)


    if isinstance(material, (
        MeshPhongMaterial,
        MeshToonMaterial,
        MeshLambertMaterial,
        MeshBasicMaterial,
        MeshStandardMaterial,
        #ShaderMaterial
        ShadowMaterial
    )):# or isinstance(mobject, SkinnedMesh):

        uniforms['viewMatrix'] = camera.matrix.inverse()

    # skinning and morph target uniforms must be set even if material didn't change
    # auto-setting of texture unit for bone and morph texture must go before other textures
    # otherwise textures used for skinning and morphing can take over texture units reserved for other material textures

    #if isinstance(mobject, SkinnedMesh):

    #    uniforms['bindMatrix'] = mobject.bindMatrix
    #    uniforms['bindMatrixInverse'] = mobject.bindMatrixInverse

    #    const skeleton = mobject.skeleton;

    #    if ( skeleton ):

    #        if ( capabilities.floatVertexTextures ):

    #            if ( skeleton.boneTexture == null ) skeleton.computeBoneTexture();

    #            uniforms['boneTexture'], skeleton.boneTexture, textures );
    #            uniforms['boneTextureSize', skeleton.boneTextureSize );

    #        } else {

    #            console.warn( 'THREE.WebGLRenderer: SkinnedMesh can only be used with WebGL 2. With WebGL 1 OES_texture_float and vertex textures support is required.' );

    #        }

    #    }

    #}

    #const morphAttributes = geometry.morphAttributes;

    #if ( morphAttributes.position !== undefined || morphAttributes.normal !== undefined || ( morphAttributes.color !== undefined && capabilities.isWebGL2 == true ) ):

    #    morphtargets.update( object, geometry, material, program );

    #}

    #if ( refreshMaterial || materialProperties.receiveShadow !== object.receiveShadow ):

    #materialProperties.receiveShadow = object.receiveShadow;
    uniforms['receiveShadow'] = mobject.receive_shadow  # bring to material?

    #}

    # https:#github.com/mrdoob/three.js/pull/24467#issuecomment-1209031512

    #if ( material.isMeshGouraudMaterial && material.envMap !== null ):

    #    m_uniforms["envMap"] = envMap;

    #    m_uniforms["flipEnvMap"] = ( envMap.isCubeTexture && envMap.isRenderTargetTexture == false ) ? - 1 : 1;

    #}

    #if ( refreshMaterial ):

    uniforms['toneMappingExposure'] = 1.0

    #if ( materialProperties.needsLights ):

    #    # the current material requires lighting info

    #    # note: all lighting uniforms are always set correctly
    #    # they simply reference the renderer's state for their
    #    # values
    #    #
    #    # use the current material's .needsUpdate flags to set
    #    # the GL state when required

    #    markUniformsLightsNeedsUpdate( m_uniforms, refreshLights );

    #}

    # refresh uniforms common to several materials

    m = convert_to_universal_material(material)
    if fog is not None and m.fog:
        uniforms["fogColor"] = fog.color

        if isinstance(fog, FogLinear):
            uniforms["fogNear"] = fog.near
            uniforms["fogFar"] = fog.far

        elif isinstance(fog, FogExp2):
            uniforms["fogDensity"] = fog.density

    _pixelRatio = 1.0
    _height = 540
    _transmissionRenderTarget = None
    refreshMaterialUniforms(uniforms, material, _pixelRatio, _height, _transmissionRenderTarget)

    #WebGLUniforms.upload( _gl, materialProperties.uniformsList, m_uniforms, textures );

    #}

    #if ( material.isShaderMaterial && material.uniformsNeedUpdate == true ):

    #    WebGLUniforms.upload( _gl, materialProperties.uniformsList, m_uniforms, textures );
    #    material.uniformsNeedUpdate = false;

    #}

    if isinstance(mobject, Sprite):
        uniforms['center'] = mobject.center

    # common matrices

    modelMatrix = mobject.matrix
    modelViewMatrix = modelMatrix.apply(camera.matrix.inverse())
    #mat = ~modelViewMatrix
    normalMatrix = Mat3.from_mat4(modelViewMatrix.inverse())# Mat3((*mat[0:3], *mat[4:7], *mat[8:11]))
    uniforms['modelViewMatrix'] = modelViewMatrix
    uniforms['normalMatrix'] = normalMatrix
    uniforms['modelMatrix'] = modelMatrix

    # UBOs

    #if ( material.isShaderMaterial || material.isRawShaderMaterial ):

    #    const groups = material.uniformsGroups;

    #    for ( let i = 0, l = groups.length; i < l; i ++ ) {

    #        if ( capabilities.isWebGL2 ):

    #            const group = groups[ i ];

    #            uniformsGroups.update( group, program );
    #            uniformsGroups.bind( group, program );

    #        } else {

    #            console.warn( 'THREE.WebGLRenderer: Uniform Buffer Objects can only be used with WebGL 2.' );

    #        }

    #    }

    #}

    #return program;


    return uniforms


def get_attributes(mobject: Mobject, geometry: Geometry) -> dict[str, Attribute]:
    result = {}
    for attribute_name, attribute in geometry.attributes.__dict__.items():  # TODO
        if attribute is None:
            if isinstance(mobject, InstancedMesh):
                if attribute_name == "instanceMatrix" and mobject.instanceMatrix:
                    attribute = mobject.instanceMatrix
                elif attribute_name == "instanceColor" and mobject.instanceColor:
                    attribute = mobject.instanceColor
                else:
                    raise ValueError(f"Attribute '{attribute_name}' undefined")
            else:
                raise ValueError(f"Attribute '{attribute_name}' undefined")
        result[attribute_name] = attribute
    return result

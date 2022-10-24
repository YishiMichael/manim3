from __future__ import annotations

from dataclasses import dataclass

from colour import Color

from constants import MultiplyOperation
from materials.material import Material, Uniform
from utils.texture import Texture


@dataclass
class MeshBasicMaterial(Material):
    color: Color = Color("#ffffff") # emissive

    map: Texture | None = None

    lightMap: Texture | None = None
    lightMapIntensity: float = 1.0

    aoMap: Texture | None = None
    aoMapIntensity: float = 1.0

    specularMap: Texture | None = None

    alphaMap: Texture | None = None

    envMap: Texture | None = None
    combine: int = MultiplyOperation
    reflectivity: float = 1.0
    refractionRatio: float = 0.98

    wireframe: bool = False
    wireframeLinewidth: float = 1.0
    wireframeLinecap: str = "round"
    wireframeLinejoin: str = "round"

    fog: bool = True

    def construct_uniform(self) -> dict[str, Uniform]:
        uniforms: dict[str, Uniform] = {}
        uniforms["opacity"] = self.opacity

        if ( self.color ):

            uniforms["diffuse"] = ( self.color )

        if ( self.emissive ):

            uniforms["emissive"] = ( self.emissive ).multiplyScalar( self.emissiveIntensity )

        if ( self.map ):

            uniforms["map"] = self.map

        if ( self.alphaMap ):

            uniforms["alphaMap"] = self.alphaMap

        if ( self.bumpMap ):

            uniforms["bumpMap"] = self.bumpMap
            uniforms["bumpScale"] = self.bumpScale
            if ( self.side == BackSide ):
                uniforms["bumpScale"] *= - 1

        if ( self.displacementMap ):

            uniforms["displacementMap"] = self.displacementMap
            uniforms["displacementScale"] = self.displacementScale
            uniforms["displacementBias"] = self.displacementBias

        if ( self.emissiveMap ):

            uniforms["emissiveMap"] = self.emissiveMap

        if ( self.normalMap ):

            uniforms["normalMap"] = self.normalMap
            uniforms["normalScale"] = ( self.normalScale )
            if ( self.side == BackSide ):
                uniforms["normalScale"].negate()

        if ( self.specularMap ):

            uniforms["specularMap"] = self.specularMap

        if ( self.alphaTest > 0 ):

            uniforms["alphaTest"] = self.alphaTest

        #envMap = None

        #if ( envMap ):

        #    uniforms["envMap"] = envMap

        #    uniforms["flipEnvMap"] = -1 if (isinstance(envMap, CubeTexture) and not envMap.isRenderTargetTexture) else 1

        #    uniforms["reflectivity"] = self.reflectivity
        #    uniforms["ior"] = self.ior
        #    uniforms["refractionRatio"] = self.refractionRatio

        if ( self.lightMap ):

            uniforms["lightMap"] = self.lightMap

            # artist-friendly light intensity scaling factor
            physicallyCorrectLights = True
            scaleFactor = 1.0 if physicallyCorrectLights else math.pi

            uniforms["lightMapIntensity"] = self.lightMapIntensity * scaleFactor

        if ( self.aoMap ):

            uniforms["aoMap"] = self.aoMap
            uniforms["aoMapIntensity"] = self.aoMapIntensity

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


        if ( self.map ):

            uvScaleMap = self.map

        elif ( self.specularMap ):

            uvScaleMap = self.specularMap

        elif ( self.displacementMap ):

            uvScaleMap = self.displacementMap

        elif ( self.normalMap ):

            uvScaleMap = self.normalMap

        elif ( self.bumpMap ):

            uvScaleMap = self.bumpMap

        elif ( self.roughnessMap ):

            uvScaleMap = self.roughnessMap

        elif ( self.metalnessMap ):

            uvScaleMap = self.metalnessMap

        elif ( self.alphaMap ):

            uvScaleMap = self.alphaMap

        elif ( self.emissiveMap ):

            uvScaleMap = self.emissiveMap

        elif ( self.clearcoatMap ):

            uvScaleMap = self.clearcoatMap

        elif ( self.clearcoatNormalMap ):

            uvScaleMap = self.clearcoatNormalMap

        elif ( self.clearcoatRoughnessMap ):

            uvScaleMap = self.clearcoatRoughnessMap

        elif ( self.iridescenceMap ):

            uvScaleMap = self.iridescenceMap

        elif ( self.iridescenceThicknessMap ):

            uvScaleMap = self.iridescenceThicknessMap

        elif ( self.specularIntensityMap ):

            uvScaleMap = self.specularIntensityMap

        elif ( self.specularColorMap ):

            uvScaleMap = self.specularColorMap

        elif ( self.transmissionMap ):

            uvScaleMap = self.transmissionMap

        elif ( self.thicknessMap ):

            uvScaleMap = self.thicknessMap

        elif ( self.sheenColorMap ):

            uvScaleMap = self.sheenColorMap

        elif ( self.sheenRoughnessMap ):

            uvScaleMap = self.sheenRoughnessMap

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

        if ( self.aoMap ):

            uv2ScaleMap = self.aoMap

        elif ( self.lightMap ):

            uv2ScaleMap = self.lightMap

        else:
            uv2ScaleMap = None

        if uv2ScaleMap is not None:
            # backwards compatibility
            #if ( uv2ScaleMap.isWebGLRenderTarget ):

            #    uv2ScaleMap = uv2ScaleMap.texture


            #if ( uv2ScaleMap.matrixAutoUpdate ):

            #    uv2ScaleMap.updateMatrix()

            uniforms["uv2Transform"] = ( uv2ScaleMap.matrix )

        return uniforms

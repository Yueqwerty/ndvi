// Advanced Cloud Masking Script for Landsat 8

// 1. Funciones auxiliares matemáticas
function abs(x) { return Math.abs(x); }
function sqrt(x) { return Math.sqrt(x); }
function log(x) { return Math.log(x); }
function exp(x) { return Math.exp(x); }

// 2. Definición avanzada de bandas
var B1_coastal = B1;    // Banda costera/aerosol
var B2_blue = B2;       // Azul
var B3_green = B3;      // Verde
var B4_red = B4;        // Rojo
var B5_nir = B5;        // Infrarrojo cercano
var B6_swir1 = B6;      // SWIR 1
var B7_swir2 = B7;      // SWIR 2
var B9_cirrus = B9;     // Cirrus
var B10_tir1 = B10;     // Térmico infrarrojo 1
var B11_tir2 = B11;     // Térmico infrarrojo 2

// 3. Función de detección de nubes multi-criterio
function advancedCloudDetection() {
    // Criterio 1: Índice de Nubes (CI - Cloud Index)
    var cloudIndex = 0;

    // Sub-criterio de Cirrus
    var cirrusCriteria = (B9_cirrus > 0.1) ? 0.3 : 0;
    cloudIndex += cirrusCriteria;

    // Sub-criterio de bandas visibles
    var visibleReflectance = (B2_blue + B3_green + B4_red) / 3;
    var visibleCriteria = (visibleReflectance > 0.4) ? 0.25 : 0;
    cloudIndex += visibleCriteria;

    // Sub-criterio de infrarrojo cercano
    var nirCloudness = 1 - (B5_nir / (B4_red + 0.1));
    var nirCriteria = (nirCloudness > 0.3) ? 0.2 : 0;
    cloudIndex += nirCriteria;

    // Sub-criterio de diferencial térmico
    var thermalDifference = abs(B10_tir1 - B11_tir2);
    var thermalCriteria = (thermalDifference > 2) ? 0.15 : 0;
    cloudIndex += thermalCriteria;

    // Sub-criterio de bandas SWIR
    var swirRatio = B6_swir1 / (B7_swir2 + 0.1);
    var swirCriteria = (abs(1 - swirRatio) > 0.2) ? 0.1 : 0;
    cloudIndex += swirCriteria;

    // 4. Clasificación de tipos de nubes
    var cloudType = 'clear';
    if (cloudIndex > 0.5) {
        // Nubes probables
        cloudType = 'probable_cloud';
    }
    if (cloudIndex > 0.8) {
        // Nubes casi seguras
        cloudType = 'high_confidence_cloud';
    }

    // 5. Análisis de sombras de nubes
    var shadowIndex = calculateCloudShadow();

    // 6. Retorno de información detallada
    return {
        cloudIndex: cloudIndex,
        cloudType: cloudType,
        shadowIndex: shadowIndex
    };
}

// 7. Función para detectar sombras de nubes
function calculateCloudShadow() {
    // Estimación de sombras basada en diferencias espectrales
    var shadowRatio = B6_swir1 / (B5_nir + 0.1);
    var shadowIntensity = log(1 + shadowRatio);
    return shadowIntensity;
}

// 8. Procesamiento final y visualización
function processImage() {
    var cloudDetection = advancedCloudDetection();

    // Colores basados en confianza de nubes
    var CLEAR = [B4, B3, B2];  // Imagen RGB normal
    var PROBABLE_CLOUD = [0.8, 0.8, 0.8];  // Gris claro
    var HIGH_CONFIDENCE_CLOUD = [1, 1, 1];  // Blanco
    var CLOUD_SHADOW = [0.3, 0.3, 0.3];  // Gris oscuro

    // Decisión final de visualización
    if (cloudDetection.cloudType === 'high_confidence_cloud') {
        return HIGH_CONFIDENCE_CLOUD;
    } else if (cloudDetection.cloudType === 'probable_cloud') {
        // Si hay sombra de nube significativa
        if (cloudDetection.shadowIndex > 0.5) {
            return CLOUD_SHADOW;
        }
        return PROBABLE_CLOUD;
    } else {
        return CLEAR;
    }
}

// 9. Ejecución final
return processImage();
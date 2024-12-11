
function abs(x) { return Math.abs(x); }
function sqrt(x) { return Math.sqrt(x); }
function log(x) { return Math.log(x); }
function exp(x) { return Math.exp(x); }

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

function advancedCloudDetection() {
    var cloudIndex = 0;

    var cirrusCriteria = (B9_cirrus > 0.1) ? 0.3 : 0;
    cloudIndex += cirrusCriteria;

    var visibleReflectance = (B2_blue + B3_green + B4_red) / 3;
    var visibleCriteria = (visibleReflectance > 0.4) ? 0.25 : 0;
    cloudIndex += visibleCriteria;

    var nirCloudness = 1 - (B5_nir / (B4_red + 0.1));
    var nirCriteria = (nirCloudness > 0.3) ? 0.2 : 0;
    cloudIndex += nirCriteria;

    var thermalDifference = abs(B10_tir1 - B11_tir2);
    var thermalCriteria = (thermalDifference > 2) ? 0.15 : 0;
    cloudIndex += thermalCriteria;

    var swirRatio = B6_swir1 / (B7_swir2 + 0.1);
    var swirCriteria = (abs(1 - swirRatio) > 0.2) ? 0.1 : 0;
    cloudIndex += swirCriteria;

    var cloudType = 'clear';
    if (cloudIndex > 0.5) {
        cloudType = 'probable_cloud';
    }
    if (cloudIndex > 0.8) {
        cloudType = 'high_confidence_cloud';
    }

    var shadowIndex = calculateCloudShadow();

    return {
        cloudIndex: cloudIndex,
        cloudType: cloudType,
        shadowIndex: shadowIndex
    };
}

function calculateCloudShadow() {
    var shadowRatio = B6_swir1 / (B5_nir + 0.1);
    var shadowIntensity = log(1 + shadowRatio);
    return shadowIntensity;
}

function processImage() {
    var cloudDetection = advancedCloudDetection();

    var CLEAR = [B4, B3, B2];  // Imagen RGB normal
    var PROBABLE_CLOUD = [0.8, 0.8, 0.8];  // Gris claro
    var HIGH_CONFIDENCE_CLOUD = [1, 1, 1];  // Blanco
    var CLOUD_SHADOW = [0.3, 0.3, 0.3];  // Gris oscuro

    if (cloudDetection.cloudType === 'high_confidence_cloud') {
        return HIGH_CONFIDENCE_CLOUD;
    } else if (cloudDetection.cloudType === 'probable_cloud') {
        if (cloudDetection.shadowIndex > 0.5) {
            return CLOUD_SHADOW;
        }
        return PROBABLE_CLOUD;
    } else {
        return CLEAR;
    }
}

return processImage();
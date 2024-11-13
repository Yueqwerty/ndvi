# run_pipeline.ps1

Param (
    [string]$FechaActual = "2024-10-21",
    [string]$FechaAnterior = "2024-10-20",
    [float]$Umbral = 0.1
)

# Función para ejecutar un comando y verificar el resultado
function Execute-Command {
    param (
        [string]$Command,
        [string]$Description
    )
    Write-Host "=== $Description ==="
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error al ejecutar: $Description. Terminando el pipeline."
        exit 1
    }
}

# Definir las fechas a procesar
$FECHAS = @($FechaActual, $FechaAnterior)

# Iterar sobre cada fecha para obtener y procesar NDVI
foreach ($FECHA in $FECHAS) {
    Execute-Command "python scripts\fetch_ndvi.py $FECHA" "Obteniendo NDVI para $FECHA"
    Execute-Command "python scripts\process_ndvi.py $FECHA" "Procesando NDVI para $FECHA"
}

# Comparar fechas especificadas
Execute-Command "python scripts\compare_ndvi.py $FechaAnterior $FechaActual" "Comparando NDVI entre $FechaAnterior y $FechaActual"

# Generar visualizaciones
Execute-Command "python scripts\visualize_ndvi.py $FechaAnterior $FechaActual" "Generando visualizaciones para $FechaAnterior y $FechaActual"

Write-Host "=== Pipeline de NDVI completado exitosamente ==="

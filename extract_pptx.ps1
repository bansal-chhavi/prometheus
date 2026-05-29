# Try using temp rename method since Expand-Archive doesn't recognize .pptx
$zipPath = 'c:\github\prometheus\KnowledgeOps_TeamSpecs.pptx'
$extractPath = 'c:\github\prometheus\pptx_extracted'
$tempZipPath = 'c:\github\prometheus\temp_archive.zip'

if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }

# Copy as .zip and extract
Copy-Item $zipPath $tempZipPath
Expand-Archive -Path $tempZipPath -DestinationPath $extractPath
Remove-Item $tempZipPath

# Read slide content
$slideFiles = Get-ChildItem "$extractPath\ppt\slides" -Filter 'slide*.xml' | Sort-Object Name

foreach ($slideFile in $slideFiles) {
    Write-Host "=== $($slideFile.Name) ===" -ForegroundColor Green
    [xml]$content = Get-Content $slideFile.FullName
    $ns = @{a='http://schemas.openxmlformats.org/drawingml/2006/main'; p='http://schemas.openxmlformats.org/presentationml/2006/main'; r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    
    $textElements = $content.SelectNodes('.//a:t', $ns)
    foreach ($textElem in $textElements) {
        Write-Host $textElem.InnerText
    }
    Write-Host ""
}

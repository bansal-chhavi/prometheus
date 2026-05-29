# Simpler extraction - just read raw XML with text nodes
$extractPath = 'c:\github\prometheus\pptx_extracted'

$slideFiles = Get-ChildItem "$extractPath\ppt\slides" -Filter 'slide*.xml' | Sort-Object {[int]($_.BaseName -replace '\D')}

foreach ($slideFile in $slideFiles) {
    Write-Host "=== $($slideFile.Name) ===" -ForegroundColor Green
    $xmlContent = Get-Content $slideFile.FullName -Raw
    
    # Extract all text nodes using regex
    $textMatches = [regex]::Matches($xmlContent, '<a:t>([^<]+)</a:t>')
    foreach ($match in $textMatches) {
        Write-Host $match.Groups[1].Value
    }
    Write-Host ""
}

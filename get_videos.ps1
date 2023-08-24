function Get-MediaDuration {
    param (
        [string]$Path
    )
    $ffprobeOutput = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $Path
    if ($ffprobeOutput) {
        $duration = [double]::Parse($ffprobeOutput, [System.Globalization.CultureInfo]::InvariantCulture)
        return $duration
    }
    return 0
}

$sourceFolder = "D:\Videos\TV"
$destinationFolder = "C:\Users\alec\source\python\depth_video\unformatted_videos"
$ffmpegPath = "ffmpeg"

$mp4Files = Get-ChildItem -Path $sourceFolder -Recurse -Filter *.mp4 | Select-Object -ExpandProperty FullName

if ($mp4Files.Count -gt 0) {
    $randomVideos = Get-Random -InputObject $mp4Files -Count 5

    foreach ($video in $randomVideos) {
        $videoName = [System.IO.Path]::GetFileNameWithoutExtension($video)
        $outputFile = Join-Path -Path $destinationFolder -ChildPath "$videoName-random-1min.mp4"
        $startTime = Get-Random -Minimum 0 -Maximum ([math]::Max(0, (Get-MediaDuration -Path $video) - 60))
        & $ffmpegPath -i $video -ss $startTime -t 60 -c:v copy -c:a copy $outputFile
    }
} else {
    Write-Host "No .mp4 files found in the specified folder."
}
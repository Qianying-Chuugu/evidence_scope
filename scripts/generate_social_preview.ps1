Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $root 'static/evidence-scope-demo.png'
$outputPath = Join-Path $root 'static/social-preview.png'

$image = [System.Drawing.Bitmap]::new(1280, 640)
$graphics = [System.Drawing.Graphics]::FromImage($image)
$source = [System.Drawing.Image]::FromFile($sourcePath)

try {
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    $background = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(12, 35, 34))
    $accent = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(122, 225, 178))
    $white = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(245, 250, 247))
    $muted = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(174, 199, 189))
    $card = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(235, 242, 234))
    $line = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(46, 88, 75), 2)
    $titleFont = [System.Drawing.Font]::new('Segoe UI', 54, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $taglineFont = [System.Drawing.Font]::new('Microsoft YaHei UI', 28, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $detailFont = [System.Drawing.Font]::new('Segoe UI', 21, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $smallFont = [System.Drawing.Font]::new('Microsoft YaHei UI', 17, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $logoFont = [System.Drawing.Font]::new('Georgia', 48, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)

    $graphics.FillRectangle($background, 0, 0, 1280, 640)
    $graphics.FillRectangle($accent, 74, 75, 62, 62)
    $graphics.DrawString('E', $logoFont, $background, 88, 74)
    $graphics.DrawString('EvidenceScope', $titleFont, $white, 69, 174)
    $graphics.DrawString('让本地 PDF 研究有据可查', $taglineFont, $white, 74, 284)
    $graphics.DrawLine($line, 75, 350, 439, 350)
    $graphics.DrawString('Agent 工具调用  /  混合检索', $smallFont, $muted, 75, 378)
    $graphics.DrawString('文档来源  /  页码  /  Chunk ID', $smallFont, $muted, 75, 411)
    $graphics.DrawString('LOCAL PDF RESEARCH AGENT', $detailFont, $accent, 75, 529)

    $graphics.FillRectangle($card, 545, 36, 681, 568)
    $graphics.DrawImage($source, [System.Drawing.Rectangle]::new(555, 46, 661, 548))
    $graphics.FillRectangle($background, 968, 556, 240, 38)
    $graphics.DrawString('真实应用界面', $smallFont, $white, 1032, 562)

    $image.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $source.Dispose()
    $graphics.Dispose()
    $image.Dispose()
    foreach ($item in @($background, $accent, $white, $muted, $card, $line, $titleFont, $taglineFont, $detailFont, $smallFont, $logoFont)) {
        if ($null -ne $item) { $item.Dispose() }
    }
}

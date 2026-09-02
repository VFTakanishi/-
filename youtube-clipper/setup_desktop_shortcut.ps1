<#
.SYNOPSIS
    デスクトップに「Podcast Clipper」ショートカットを1つ作成します。

.DESCRIPTION
    これは一度だけ実行してください。デスクトップに .lnk ファイルが1つ
    できるだけで、他には何も変更しません（Windowsのスタートアップ登録・
    タスクスケジューラ・レジストリ・サービス登録は一切行いません）。

    以降は、そのショートカットをダブルクリックするだけで:
      - 黒いコマンドプロンプト画面を表示せずに（pythonw.exe を使用）
      - 裏でPodcast Clipperのサーバーが起動し
      - 既定のブラウザで自動的に開きます

    既に起動中の場合（ショートカットを2回押した場合など）は、
    launch.py 側の重複起動チェックにより新しいサーバーは起動されず、
    既存のものがブラウザで開きます。

    起動時のメッセージ・エラーは画面には出ず、
    youtube-clipper\logs\launcher.log に記録されます。

    元に戻すには、作成されたショートカットをデスクトップから削除する
    だけです。それ以外に変更された設定はありません。

    start.bat（従来の起動方法）はこれまでどおり使えます。
#>

$ErrorActionPreference = "Stop"

$repoDir = $PSScriptRoot
$launchScript = Join-Path $repoDir "launch.py"

if (-not (Test-Path $launchScript)) {
    Write-Host "エラー: launch.py が見つかりません（$launchScript）。" -ForegroundColor Red
    Write-Host "このスクリプトは youtube-clipper フォルダの中で実行してください。"
    exit 1
}

# start.bat と同じ優先順位で pythonw.exe（コンソールを開かない版）を探す:
# .venv 内 > PATH上の python と同じフォルダ > PATH上の py ランチャーが指す実体
$pythonwPath = $null

$venvPythonw = Join-Path $repoDir ".venv\Scripts\pythonw.exe"
if (Test-Path $venvPythonw) {
    $pythonwPath = $venvPythonw
}

if (-not $pythonwPath) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidate = Join-Path (Split-Path $pythonCmd.Source) "pythonw.exe"
        if (Test-Path $candidate) {
            $pythonwPath = $candidate
        }
    }
}

if (-not $pythonwPath) {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $resolved = & py -3 -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))" 2>$null
        if ($resolved -and (Test-Path $resolved)) {
            $pythonwPath = $resolved
        }
    }
}

if (-not $pythonwPath) {
    Write-Host "エラー: pythonw.exe が見つかりませんでした。" -ForegroundColor Red
    Write-Host ""
    Write-Host "先に README.md の手順で Python 環境（.venv 推奨）をセットアップしてから、"
    Write-Host "このスクリプトをもう一度実行してください。"
    exit 1
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Podcast Clipper.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonwPath
$shortcut.Arguments = "launch.py"
$shortcut.WorkingDirectory = $repoDir
$shortcut.IconLocation = "$pythonwPath,0"
$shortcut.Description = "Podcast Clipper を起動します（黒い画面は表示されません）"
$shortcut.Save()

Write-Host "デスクトップに『Podcast Clipper』ショートカットを作成しました。" -ForegroundColor Green
Write-Host "  $shortcutPath"
Write-Host ""
Write-Host "使用したPython: $pythonwPath"
Write-Host ""
Write-Host "今後はこのショートカットをダブルクリックするだけで起動できます。"
Write-Host "元に戻すには、このショートカットを削除するだけです。"

# Moon

A Direct3D 12 demo application for Windows.

## Features

- **Rotating sphere** with a texture loaded from `moon.png` via WIC (Windows
  Imaging Component).  If `moon.png` is not present the sphere is rendered in
  solid blue as a fallback.
- **Checkerboard plane** rendered behind the sphere using a fully procedural
  pixel shader (no texture asset required).
- **Direct2D overlay** rendered on top via the D3D11On12 interoperability
  layer: a yellow rectangle containing the word **"Moon"** in Segoe UI font.

## Requirements

- Windows 10 / 11 with a DirectX 12-capable GPU (or WARP software renderer)
- Visual Studio 2019 or 2022 (v142 / v143 platform toolset)
- Windows SDK 10.0 or later

## Build

Open `Moon.sln` in Visual Studio and build the **Release x64** (or
**Debug x64**) configuration.

## Running

Place a `moon.png` image in the same directory as the compiled executable
(`Moon.exe`) before launching.  The sphere will be textured with that image.
Press **Escape** to exit.

## File overview

| File | Purpose |
|---|---|
| `Main.cpp` | Entire application: D3D12 init, WIC loading, geometry, render loop, D2D overlay |
| `Moon.vcxproj` | Visual Studio project (x64, C++17, Windows subsystem) |
| `Moon.sln` | Visual Studio solution |

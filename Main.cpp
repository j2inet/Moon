// =============================================================================
// Main.cpp  -  Direct3D 12 Moon Demo
//
// Features:
//   * WIC-loaded moon.png texture mapped onto a UV sphere
//   * Sphere rotates continuously on the Y axis
//   * Checkerboard plane rendered behind/below the sphere
//   * Direct2D overlay (via D3D11On12): yellow rectangle + "Moon" in Segoe UI
// =============================================================================

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <wincodec.h>
#include <d2d1_3.h>
#include <d3d11on12.h>
#include <dwrite.h>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")

using namespace DirectX;
using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr UINT FrameCount    = 2;
static constexpr UINT WinWidth      = 1280;
static constexpr UINT WinHeight     = 720;
static constexpr UINT SphereStacks  = 30;
static constexpr UINT SphereSectors = 50;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct Vertex
{
    XMFLOAT3 pos;
    XMFLOAT3 normal;
    XMFLOAT2 uv;
};

// Each constant-buffer slot must be 256-byte aligned for root CBVs.
// 2 x float4x4 = 128 B; 32 floats padding = 128 B; total = 256 B.
struct alignas(256) ObjectConstants
{
    XMFLOAT4X4 wvp;
    XMFLOAT4X4 world;
    float       _pad[32];
};
static_assert(sizeof(ObjectConstants) == 256, "ObjectConstants must be 256 bytes");

// ---------------------------------------------------------------------------
// Inline HLSL shaders
// ---------------------------------------------------------------------------
static const char* VS_SRC = R"hlsl(
cbuffer ObjCB : register(b0)
{
    float4x4 gWVP;
    float4x4 gWorld;
};
struct VIn  { float3 pos : POSITION; float3 nor : NORMAL; float2 uv : TEXCOORD; };
struct VOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };
VOut main(VIn v)
{
    VOut o;
    o.pos = mul(float4(v.pos, 1.0f), gWVP);
    o.uv  = v.uv;
    return o;
}
)hlsl";

static const char* PS_TEXTURED_SRC = R"hlsl(
Texture2D    gTex : register(t0);
SamplerState gSmp : register(s0);
struct PIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };
float4 main(PIn p) : SV_Target
{
    return gTex.Sample(gSmp, p.uv);
}
)hlsl";

static const char* PS_CHECKER_SRC = R"hlsl(
struct PIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD; };
float4 main(PIn p) : SV_Target
{
    int2  c = int2(floor(p.uv * 10.0f));
    float t = ((c.x + c.y) & 1) ? 0.85f : 0.15f;
    return float4(t, t, t, 1.0f);
}
)hlsl";

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
HWND g_hwnd;

// D3D12 core
ComPtr<ID3D12Device>              g_dev;
ComPtr<ID3D12CommandQueue>        g_queue;
ComPtr<IDXGISwapChain3>           g_sc;
ComPtr<ID3D12CommandAllocator>    g_alloc[FrameCount];
ComPtr<ID3D12CommandAllocator>    g_initAlloc;
ComPtr<ID3D12GraphicsCommandList> g_cl;
ComPtr<ID3D12Fence>               g_fence;
HANDLE                            g_fenceEvent = nullptr;
UINT64                            g_fenceVal   = 0;
UINT64                            g_frameFence[FrameCount] = {};
UINT                              g_frame      = 0;
UINT                              g_rtvInc     = 0;
UINT                              g_srvInc     = 0;

// Descriptor heaps
ComPtr<ID3D12DescriptorHeap> g_rtvHeap;
ComPtr<ID3D12DescriptorHeap> g_dsvHeap;
ComPtr<ID3D12DescriptorHeap> g_srvHeap;

// Back buffers + depth
ComPtr<ID3D12Resource> g_bb[FrameCount];
ComPtr<ID3D12Resource> g_depth;

// Root signatures & PSOs
ComPtr<ID3D12RootSignature> g_rsTextured;
ComPtr<ID3D12RootSignature> g_rsChecker;
ComPtr<ID3D12PipelineState> g_psoTextured;
ComPtr<ID3D12PipelineState> g_psoChecker;

// Sphere geometry
ComPtr<ID3D12Resource>   g_sphereVB;
ComPtr<ID3D12Resource>   g_sphereIB;
D3D12_VERTEX_BUFFER_VIEW g_sphereVBV = {};
D3D12_INDEX_BUFFER_VIEW  g_sphereIBV = {};
UINT                     g_sphereIdxCount = 0;

// Plane geometry
ComPtr<ID3D12Resource>   g_planeVB;
ComPtr<ID3D12Resource>   g_planeIB;
D3D12_VERTEX_BUFFER_VIEW g_planeVBV = {};
D3D12_INDEX_BUFFER_VIEW  g_planeIBV = {};
UINT                     g_planeIdxCount = 0;

// Moon texture
ComPtr<ID3D12Resource> g_moonTex;

// Constant buffer - persistent-mapped, 4 slots x 256 B
ComPtr<ID3D12Resource> g_cb;
UINT8*                 g_cbPtr = nullptr;

// Upload helpers kept alive until init GPU work completes
std::vector<ComPtr<ID3D12Resource>> g_uploads;

// D3D11On12 / D2D
ComPtr<ID3D11On12Device>     g_11on12;
ComPtr<ID3D11DeviceContext>  g_d3d11Ctx;
ComPtr<ID2D1Factory3>        g_d2dFac;
ComPtr<ID2D1Device2>         g_d2dDev;
ComPtr<ID2D1DeviceContext2>  g_d2dCtx;
ComPtr<IDWriteFactory>       g_dwFac;
ComPtr<IDWriteTextFormat>    g_textFmt;
ComPtr<ID2D1SolidColorBrush> g_yellowBrush;
ComPtr<ID2D1SolidColorBrush> g_blackBrush;
ComPtr<ID3D11Resource>       g_wrapped[FrameCount];
ComPtr<ID2D1Bitmap1>         g_d2dRT[FrameCount];

// Animation
float         g_angle   = 0.0f;
LARGE_INTEGER g_prevTime{};
LARGE_INTEGER g_freq{};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void HR(HRESULT hr)
{
    if (FAILED(hr))
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "HRESULT 0x%08X", static_cast<unsigned>(hr));
        throw std::runtime_error(buf);
    }
}

static ComPtr<ID3DBlob> CompileShader(const char* src, const char* ep,
                                      const char* target)
{
    ComPtr<ID3DBlob> blob, err;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
    flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
    HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr,
                            ep, target, flags, 0, &blob, &err);
    if (FAILED(hr) && err)
        OutputDebugStringA(static_cast<char*>(err->GetBufferPointer()));
    HR(hr);
    return blob;
}

static ComPtr<ID3D12Resource> MakeBuffer(UINT64 size,
                                         D3D12_HEAP_TYPE heap,
                                         D3D12_RESOURCE_STATES state,
                                         D3D12_RESOURCE_FLAGS flags =
                                             D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = size;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = flags;
    ComPtr<ID3D12Resource> res;
    HR(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                      state, nullptr, IID_PPV_ARGS(&res)));
    return res;
}

// Upload CPU data to a GPU-only buffer, recording copy into the open g_cl.
static ComPtr<ID3D12Resource> UploadData(const void* data, UINT64 size,
                                         D3D12_RESOURCE_STATES finalState)
{
    auto def = MakeBuffer(size, D3D12_HEAP_TYPE_DEFAULT,
                          D3D12_RESOURCE_STATE_COPY_DEST);
    auto upl = MakeBuffer(size, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    upl->Map(0, nullptr, &mapped);
    memcpy(mapped, data, static_cast<size_t>(size));
    upl->Unmap(0, nullptr);
    g_cl->CopyBufferRegion(def.Get(), 0, upl.Get(), 0, size);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = def.Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter  = finalState;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &b);
    g_uploads.push_back(upl);
    return def;
}

// ---------------------------------------------------------------------------
// Geometry generators
// ---------------------------------------------------------------------------
static void GenSphere(float r, UINT stacks, UINT sectors,
                      std::vector<Vertex>& verts, std::vector<UINT>& idx)
{
    for (UINT i = 0; i <= stacks; ++i)
    {
        float phi = static_cast<float>(i) / stacks * XM_PI;
        float sp  = sinf(phi);
        float cp  = cosf(phi);
        for (UINT j = 0; j <= sectors; ++j)
        {
            float theta = static_cast<float>(j) / sectors * XM_2PI;
            Vertex v;
            v.pos    = { r * sp * cosf(theta), r * cp, r * sp * sinf(theta) };
            v.normal = {     sp * cosf(theta),     cp,     sp * sinf(theta)  };
            v.uv     = { static_cast<float>(j) / sectors,
                         static_cast<float>(i) / stacks };
            verts.push_back(v);
        }
    }
    for (UINT i = 0; i < stacks; ++i)
    {
        for (UINT j = 0; j < sectors; ++j)
        {
            UINT a = i * (sectors + 1) + j;
            UINT b = a + sectors + 1;
            if (i != 0)
            {
                idx.push_back(a);
                idx.push_back(a + 1);
                idx.push_back(b);
            }
            if (i + 1 != stacks)
            {
                idx.push_back(a + 1);
                idx.push_back(b + 1);
                idx.push_back(b);
            }
        }
    }
}

static void GenPlane(float hw, float hd,
                     std::vector<Vertex>& verts, std::vector<UINT>& idx)
{
    verts = {
        { { -hw, 0.0f, -hd }, { 0,1,0 }, { 0,0 } },
        { {  hw, 0.0f, -hd }, { 0,1,0 }, { 1,0 } },
        { {  hw, 0.0f,  hd }, { 0,1,0 }, { 1,1 } },
        { { -hw, 0.0f,  hd }, { 0,1,0 }, { 0,1 } },
    };
    idx = { 0, 1, 2,  0, 2, 3 };
}

// ---------------------------------------------------------------------------
// WIC texture loader - records GPU upload commands into open g_cl.
// ---------------------------------------------------------------------------
static bool LoadMoonTexture(const wchar_t* path)
{
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    ComPtr<IWICImagingFactory> wic;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))))
        return false;
    ComPtr<IWICBitmapDecoder> dec;
    if (FAILED(wic->CreateDecoderFromFilename(path, nullptr, GENERIC_READ,
                                              WICDecodeMetadataCacheOnLoad,
                                              &dec)))
        return false;
    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return false;
    ComPtr<IWICFormatConverter> conv;
    wic->CreateFormatConverter(&conv);
    if (FAILED(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
                                WICBitmapDitherTypeNone, nullptr, 0.0,
                                WICBitmapPaletteTypeCustom)))
        return false;
    UINT W = 0, H = 0;
    conv->GetSize(&W, &H);
    UINT rowPitch  = W * 4;
    UINT imageSize = rowPitch * H;
    std::vector<UINT8> pixels(imageSize);
    conv->CopyPixels(nullptr, rowPitch, imageSize, pixels.data());

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {};
    td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width            = W;
    td.Height           = H;
    td.DepthOrArraySize = 1;
    td.MipLevels        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    HR(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                      D3D12_RESOURCE_STATE_COPY_DEST,
                                      nullptr, IID_PPV_ARGS(&g_moonTex)));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT   numRows  = 0;
    UINT64 rowBytes = 0;
    UINT64 total    = 0;
    g_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, &numRows, &rowBytes, &total);
    auto upl = MakeBuffer(total, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ);
    UINT8* mapped = nullptr;
    upl->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    for (UINT row = 0; row < numRows; ++row)
        memcpy(mapped + fp.Offset + row * fp.Footprint.RowPitch,
               pixels.data() + row * rowPitch, rowPitch);
    upl->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource        = g_moonTex.Get();
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource       = upl.Get();
    src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER b = {};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = g_moonTex.Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &b);
    g_uploads.push_back(upl);
    return true;
}

// Solid-blue fallback when moon.png is absent.
static void MakeFallbackTexture()
{
    const UINT W = 256, H = 256;
    std::vector<UINT8> px(W * H * 4);
    for (UINT i = 0; i < W * H; ++i)
    { px[i*4+0]=20; px[i*4+1]=40; px[i*4+2]=160; px[i*4+3]=255; }

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {};
    td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width            = W; td.Height = H;
    td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    HR(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                      D3D12_RESOURCE_STATE_COPY_DEST,
                                      nullptr, IID_PPV_ARGS(&g_moonTex)));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT   numRows  = 0;
    UINT64 rowBytes = 0;
    UINT64 total    = 0;
    g_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, &numRows, &rowBytes, &total);
    auto upl = MakeBuffer(total, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ);
    UINT8* mapped = nullptr;
    upl->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    for (UINT row = 0; row < numRows; ++row)
        memcpy(mapped + fp.Offset + row * fp.Footprint.RowPitch,
               px.data() + row * W * 4, W * 4);
    upl->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = g_moonTex.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = upl.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = g_moonTex.Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &b);
    g_uploads.push_back(upl);
}

// ---------------------------------------------------------------------------
// GPU fence helpers
// ---------------------------------------------------------------------------
static void WaitGPU()
{
    UINT64 v = ++g_fenceVal;
    g_queue->Signal(g_fence.Get(), v);
    if (g_fence->GetCompletedValue() < v)
    {
        g_fence->SetEventOnCompletion(v, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

static void MoveToNextFrame()
{
    g_frameFence[g_frame] = ++g_fenceVal;
    g_queue->Signal(g_fence.Get(), g_frameFence[g_frame]);
    g_frame = g_sc->GetCurrentBackBufferIndex();
    if (g_fence->GetCompletedValue() < g_frameFence[g_frame])
    {
        g_fence->SetEventOnCompletion(g_frameFence[g_frame], g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
static void InitDeviceQueue()
{
#ifdef _DEBUG
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
            dbg->EnableDebugLayer();
    }
#endif
    ComPtr<IDXGIFactory6> fac;
    HR(CreateDXGIFactory2(
#ifdef _DEBUG
        DXGI_CREATE_FACTORY_DEBUG,
#else
        0,
#endif
        IID_PPV_ARGS(&fac)));

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;
         fac->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                         IID_PPV_ARGS(&adapter)) !=
             DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(),
                                        D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&g_dev))))
            break;
        adapter.Reset();
    }
    if (!g_dev)
    {
        ComPtr<IDXGIAdapter> warp;
        fac->EnumWarpAdapter(IID_PPV_ARGS(&warp));
        HR(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0,
                             IID_PPV_ARGS(&g_dev)));
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HR(g_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue)));
    HR(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    g_rtvInc = g_dev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    g_srvInc = g_dev->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (UINT i = 0; i < FrameCount; ++i)
        HR(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         IID_PPV_ARGS(&g_alloc[i])));
    HR(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     IID_PPV_ARGS(&g_initAlloc)));
    // Leave command list open for initialization commands.
    HR(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                g_initAlloc.Get(), nullptr,
                                IID_PPV_ARGS(&g_cl)));
}

static void InitSwapChain()
{
    ComPtr<IDXGIFactory6> fac;
    HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)));
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.BufferCount      = FrameCount;
    sd.Width            = WinWidth;
    sd.Height           = WinHeight;
    sd.Format           = DXGI_FORMAT_B8G8R8A8_UNORM; // BGRA for D2D compatibility
    sd.BufferUsage      = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.SwapEffect       = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.SampleDesc.Count = 1;
    ComPtr<IDXGISwapChain1> sc1;
    HR(fac->CreateSwapChainForHwnd(g_queue.Get(), g_hwnd, &sd,
                                   nullptr, nullptr, &sc1));
    HR(sc1.As(&g_sc));
    g_frame = g_sc->GetCurrentBackBufferIndex();
}

static void InitHeapsAndBuffers()
{
    // RTV heap
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = FrameCount;
        HR(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_rtvHeap)));
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH =
            g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < FrameCount; ++i)
        {
            HR(g_sc->GetBuffer(i, IID_PPV_ARGS(&g_bb[i])));
            g_dev->CreateRenderTargetView(g_bb[i].Get(), nullptr, rtvH);
            rtvH.ptr += g_rtvInc;
        }
    }
    // DSV heap + depth buffer
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        hd.NumDescriptors = 1;
        HR(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_dsvHeap)));

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC dd = {};
        dd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dd.Width            = WinWidth;
        dd.Height           = WinHeight;
        dd.DepthOrArraySize = 1;
        dd.MipLevels        = 1;
        dd.Format           = DXGI_FORMAT_D32_FLOAT;
        dd.SampleDesc.Count = 1;
        dd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_CLEAR_VALUE cv = {};
        cv.Format              = DXGI_FORMAT_D32_FLOAT;
        cv.DepthStencil.Depth  = 1.0f;
        HR(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &dd,
                                          D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                          &cv, IID_PPV_ARGS(&g_depth)));
        D3D12_DEPTH_STENCIL_VIEW_DESC dv = {};
        dv.Format        = DXGI_FORMAT_D32_FLOAT;
        dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        g_dev->CreateDepthStencilView(
            g_depth.Get(), &dv,
            g_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    }
    // SRV heap (1 slot: moon texture)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 1;
        hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        HR(g_dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g_srvHeap)));
    }
}

static void InitRootSigs()
{
    // Textured: root CBV(b0) + SRV table(t0) + static sampler(s0)
    {
        D3D12_DESCRIPTOR_RANGE range = {};
        range.RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors     = 1;
        range.BaseShaderRegister = 0;
        range.OffsetInDescriptorsFromTableStart = 0;

        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_VERTEX;
        params[1].ParameterType                       =
            D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 1;
        params[1].DescriptorTable.pDescriptorRanges   = &range;
        params[1].ShaderVisibility                    =
            D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_STATIC_SAMPLER_DESC ss = {};
        ss.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        ss.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        ss.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        ss.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        ss.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
        ss.MaxLOD           = D3D12_FLOAT32_MAX;
        ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rd = {};
        rd.NumParameters     = 2;
        rd.pParameters       = params;
        rd.NumStaticSamplers = 1;
        rd.pStaticSamplers   = &ss;
        rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> s, e;
        HR(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                       &s, &e));
        HR(g_dev->CreateRootSignature(0, s->GetBufferPointer(),
                                      s->GetBufferSize(),
                                      IID_PPV_ARGS(&g_rsTextured)));
    }
    // Checkerboard: root CBV(b0) only
    {
        D3D12_ROOT_PARAMETER params[1] = {};
        params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_VERTEX;

        D3D12_ROOT_SIGNATURE_DESC rd = {};
        rd.NumParameters = 1;
        rd.pParameters   = params;
        rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> s, e;
        HR(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                       &s, &e));
        HR(g_dev->CreateRootSignature(0, s->GetBufferPointer(),
                                      s->GetBufferSize(),
                                      IID_PPV_ARGS(&g_rsChecker)));
    }
}

static void InitPSOs()
{
    auto vs  = CompileShader(VS_SRC,          "main", "vs_5_0");
    auto psT = CompileShader(PS_TEXTURED_SRC, "main", "ps_5_0");
    auto psC = CompileShader(PS_CHECKER_SRC,  "main", "ps_5_0");

    D3D12_INPUT_ELEMENT_DESC il[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0,
          D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
          D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24,
          D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };

    D3D12_RASTERIZER_DESC rast = {};
    rast.FillMode        = D3D12_FILL_MODE_SOLID;
    rast.CullMode        = D3D12_CULL_MODE_BACK;
    rast.DepthClipEnable = TRUE;

    D3D12_BLEND_DESC blend = {};
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    D3D12_DEPTH_STENCIL_DESC ds = {};
    ds.DepthEnable    = TRUE;
    ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    ds.DepthFunc      = D3D12_COMPARISON_FUNC_LESS;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
    pd.InputLayout           = { il, _countof(il) };
    pd.VS                    = { vs->GetBufferPointer(),  vs->GetBufferSize()  };
    pd.RasterizerState       = rast;
    pd.BlendState            = blend;
    pd.DepthStencilState     = ds;
    pd.SampleMask            = UINT_MAX;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets      = 1;
    pd.RTVFormats[0]         = DXGI_FORMAT_B8G8R8A8_UNORM;
    pd.DSVFormat             = DXGI_FORMAT_D32_FLOAT;
    pd.SampleDesc.Count      = 1;

    pd.pRootSignature = g_rsTextured.Get();
    pd.PS = { psT->GetBufferPointer(), psT->GetBufferSize() };
    HR(g_dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&g_psoTextured)));

    pd.pRootSignature = g_rsChecker.Get();
    pd.PS = { psC->GetBufferPointer(), psC->GetBufferSize() };
    HR(g_dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&g_psoChecker)));
}

static void InitGeometry()
{
    {
        std::vector<Vertex> v; std::vector<UINT> idx;
        GenSphere(1.0f, SphereStacks, SphereSectors, v, idx);
        g_sphereIdxCount = static_cast<UINT>(idx.size());
        g_sphereVB = UploadData(v.data(), v.size() * sizeof(Vertex),
                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        g_sphereIB = UploadData(idx.data(), idx.size() * sizeof(UINT),
                                D3D12_RESOURCE_STATE_INDEX_BUFFER);
        g_sphereVBV = { g_sphereVB->GetGPUVirtualAddress(),
                        static_cast<UINT>(v.size() * sizeof(Vertex)),
                        sizeof(Vertex) };
        g_sphereIBV = { g_sphereIB->GetGPUVirtualAddress(),
                        static_cast<UINT>(idx.size() * sizeof(UINT)),
                        DXGI_FORMAT_R32_UINT };
    }
    {
        std::vector<Vertex> v; std::vector<UINT> idx;
        GenPlane(8.0f, 8.0f, v, idx);
        g_planeIdxCount = static_cast<UINT>(idx.size());
        g_planeVB = UploadData(v.data(), v.size() * sizeof(Vertex),
                               D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        g_planeIB = UploadData(idx.data(), idx.size() * sizeof(UINT),
                               D3D12_RESOURCE_STATE_INDEX_BUFFER);
        g_planeVBV = { g_planeVB->GetGPUVirtualAddress(),
                       static_cast<UINT>(v.size() * sizeof(Vertex)),
                       sizeof(Vertex) };
        g_planeIBV = { g_planeIB->GetGPUVirtualAddress(),
                       static_cast<UINT>(idx.size() * sizeof(UINT)),
                       DXGI_FORMAT_R32_UINT };
    }
}

static void InitTextures()
{
    if (!LoadMoonTexture(L"moon.png"))
        MakeFallbackTexture();

    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sv.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels     = 1;
    g_dev->CreateShaderResourceView(
        g_moonTex.Get(), &sv,
        g_srvHeap->GetCPUDescriptorHandleForHeapStart());
}

static void InitConstantBuffer()
{
    // 4 slots: 2 objects x 2 frames, each 256 bytes.
    g_cb = MakeBuffer(4 * sizeof(ObjectConstants),
                      D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ);
    g_cb->Map(0, nullptr, reinterpret_cast<void**>(&g_cbPtr));
}

static void InitD2D()
{
    IUnknown* queues[] = { g_queue.Get() };
    ComPtr<ID3D11Device> d11dev;
    HR(D3D11On12CreateDevice(
        g_dev.Get(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0,
        queues, 1, 0,
        &d11dev, &g_d3d11Ctx, nullptr));
    HR(d11dev.As(&g_11on12));

    ComPtr<IDXGIDevice> dxgi;
    HR(d11dev.As(&dxgi));
    HR(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED,
                         IID_PPV_ARGS(&g_d2dFac)));
    HR(g_d2dFac->CreateDevice(dxgi.Get(), &g_d2dDev));
    HR(g_d2dDev->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                                     &g_d2dCtx));

    HR(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED,
                           __uuidof(IDWriteFactory),
                           reinterpret_cast<IUnknown**>(g_dwFac.GetAddressOf())));
    HR(g_dwFac->CreateTextFormat(
        L"Segoe UI", nullptr,
        DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL, 36.0f, L"en-us", &g_textFmt));
    g_textFmt->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);
    g_textFmt->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);

    HR(g_d2dCtx->CreateSolidColorBrush(
        D2D1::ColorF(1.0f, 1.0f, 0.0f), &g_yellowBrush));
    HR(g_d2dCtx->CreateSolidColorBrush(
        D2D1::ColorF(0.0f, 0.0f, 0.0f), &g_blackBrush));

    D2D1_BITMAP_PROPERTIES1 bp = {};
    bp.pixelFormat   = D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN,
                                         D2D1_ALPHA_MODE_PREMULTIPLIED);
    bp.dpiX = bp.dpiY = 96.0f;
    bp.bitmapOptions  = D2D1_BITMAP_OPTIONS_TARGET |
                        D2D1_BITMAP_OPTIONS_CANNOT_DRAW;

    for (UINT i = 0; i < FrameCount; ++i)
    {
        D3D11_RESOURCE_FLAGS f11 = { D3D11_BIND_RENDER_TARGET };
        HR(g_11on12->CreateWrappedResource(
            g_bb[i].Get(), &f11,
            D3D12_RESOURCE_STATE_RENDER_TARGET,  // InState
            D3D12_RESOURCE_STATE_PRESENT,         // OutState
            IID_PPV_ARGS(&g_wrapped[i])));
        ComPtr<IDXGISurface> surf;
        HR(g_wrapped[i].As(&surf));
        HR(g_d2dCtx->CreateBitmapFromDxgiSurface(surf.Get(), &bp,
                                                  &g_d2dRT[i]));
    }
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------
static void Update()
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    float dt = static_cast<float>(now.QuadPart - g_prevTime.QuadPart)
               / static_cast<float>(g_freq.QuadPart);
    g_prevTime = now;
    g_angle += dt * 0.5f;
    if (g_angle > XM_2PI) g_angle -= XM_2PI;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
static void Render()
{
    XMMATRIX view = XMMatrixLookAtLH(
        XMVectorSet(0.0f, 1.5f, 6.0f, 1.0f),
        XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f),
        XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
    XMMATRIX proj = XMMatrixPerspectiveFovLH(
        XMConvertToRadians(45.0f),
        static_cast<float>(WinWidth) / WinHeight,
        0.1f, 100.0f);
    XMMATRIX vp         = XMMatrixMultiply(view, proj);
    XMMATRIX sphereWorld = XMMatrixRotationY(g_angle);
    XMMATRIX planeWorld  = XMMatrixTranslation(0.0f, -1.5f, 0.0f);

    auto writeCB = [&](UINT obj, XMMATRIX world)
    {
        UINT64 off = static_cast<UINT64>(g_frame * 2 + obj) *
                     sizeof(ObjectConstants);
        ObjectConstants cb = {};
        XMStoreFloat4x4(&cb.wvp,   XMMatrixTranspose(world * vp));
        XMStoreFloat4x4(&cb.world, XMMatrixTranspose(world));
        memcpy(g_cbPtr + off, &cb, sizeof(ObjectConstants));
    };
    writeCB(0, sphereWorld);
    writeCB(1, planeWorld);

    HR(g_alloc[g_frame]->Reset());
    HR(g_cl->Reset(g_alloc[g_frame].Get(), nullptr));

    // Transition back buffer PRESENT -> RENDER_TARGET.
    // D3D11On12 will transition it back to PRESENT when releasing.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = g_bb[g_frame].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_cl->ResourceBarrier(1, &barrier);

    D3D12_VIEWPORT viewport = { 0, 0,
                                static_cast<float>(WinWidth),
                                static_cast<float>(WinHeight), 0.0f, 1.0f };
    D3D12_RECT sc = { 0, 0,
                      static_cast<LONG>(WinWidth),
                      static_cast<LONG>(WinHeight) };
    g_cl->RSSetViewports(1, &viewport);
    g_cl->RSSetScissorRects(1, &sc);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvH =
        g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvH.ptr += g_frame * g_rtvInc;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH =
        g_dsvHeap->GetCPUDescriptorHandleForHeapStart();

    float clearColor[] = { 0.05f, 0.05f, 0.15f, 1.0f };
    g_cl->ClearRenderTargetView(rtvH, clearColor, 0, nullptr);
    g_cl->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0,
                                0, nullptr);
    g_cl->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);

    UINT64 cbBase = g_cb->GetGPUVirtualAddress();

    // Draw checkerboard plane (behind the sphere)
    g_cl->SetGraphicsRootSignature(g_rsChecker.Get());
    g_cl->SetPipelineState(g_psoChecker.Get());
    g_cl->SetGraphicsRootConstantBufferView(
        0, cbBase + static_cast<UINT64>(g_frame * 2 + 1) *
               sizeof(ObjectConstants));
    g_cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_cl->IASetVertexBuffers(0, 1, &g_planeVBV);
    g_cl->IASetIndexBuffer(&g_planeIBV);
    g_cl->DrawIndexedInstanced(g_planeIdxCount, 1, 0, 0, 0);

    // Draw rotating sphere with moon texture
    g_cl->SetGraphicsRootSignature(g_rsTextured.Get());
    g_cl->SetPipelineState(g_psoTextured.Get());
    g_cl->SetGraphicsRootConstantBufferView(
        0, cbBase + static_cast<UINT64>(g_frame * 2 + 0) *
               sizeof(ObjectConstants));
    g_cl->SetDescriptorHeaps(1, g_srvHeap.GetAddressOf());
    g_cl->SetGraphicsRootDescriptorTable(
        1, g_srvHeap->GetGPUDescriptorHandleForHeapStart());
    g_cl->IASetVertexBuffers(0, 1, &g_sphereVBV);
    g_cl->IASetIndexBuffer(&g_sphereIBV);
    g_cl->DrawIndexedInstanced(g_sphereIdxCount, 1, 0, 0, 0);

    // Submit D3D12 commands (back buffer remains in RENDER_TARGET state).
    HR(g_cl->Close());
    ID3D12CommandList* lists[] = { g_cl.Get() };
    g_queue->ExecuteCommandLists(1, lists);

    // D2D overlay via D3D11On12
    {
        ID3D11Resource* wr = g_wrapped[g_frame].Get();
        g_11on12->AcquireWrappedResources(&wr, 1);
        g_d2dCtx->SetTarget(g_d2dRT[g_frame].Get());
        g_d2dCtx->BeginDraw();

        // Yellow filled rectangle
        D2D1_RECT_F rect = D2D1::RectF(20.0f, 20.0f, 220.0f, 80.0f);
        g_d2dCtx->FillRectangle(rect, g_yellowBrush.Get());

        // "Moon" text centered inside the rectangle
        g_d2dCtx->DrawText(L"Moon", 4, g_textFmt.Get(), rect,
                           g_blackBrush.Get());

        g_d2dCtx->EndDraw();
        g_11on12->ReleaseWrappedResources(&wr, 1);
        g_d3d11Ctx->Flush();
    }

    HR(g_sc->Present(1, 0));
    MoveToNextFrame();
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
static void Cleanup()
{
    WaitGPU();
    if (g_cbPtr)      g_cb->Unmap(0, nullptr);
    if (g_fenceEvent) CloseHandle(g_fenceEvent);
}

// ---------------------------------------------------------------------------
// Window procedure
// ---------------------------------------------------------------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (wp == VK_ESCAPE) DestroyWindow(hwnd);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// ---------------------------------------------------------------------------
// WinMain
// ---------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nCmdShow)
{
    SetProcessDPIAware();

    WNDCLASSEXW wc = {};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInst;
    wc.lpszClassName = L"MoonDemoClass";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassExW(&wc);

    RECT r = { 0, 0,
               static_cast<LONG>(WinWidth),
               static_cast<LONG>(WinHeight) };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(
        0, L"MoonDemoClass", L"Moon Demo",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        r.right - r.left, r.bottom - r.top,
        nullptr, nullptr, hInst, nullptr);

    try
    {
        QueryPerformanceFrequency(&g_freq);
        QueryPerformanceCounter(&g_prevTime);

        InitDeviceQueue();     // device, queue, cmd list (left open)
        InitSwapChain();       // DXGI swap chain
        InitHeapsAndBuffers(); // RTV/DSV heaps, depth buffer, SRV heap
        InitRootSigs();        // root signatures
        InitPSOs();            // pipeline state objects
        InitGeometry();        // sphere + plane VB/IB (records upload cmds)
        InitTextures();        // WIC / fallback texture (records upload cmds)
        InitConstantBuffer();  // persistent-mapped upload CB

        HR(g_cl->Close());
        ID3D12CommandList* initLists[] = { g_cl.Get() };
        g_queue->ExecuteCommandLists(1, initLists);
        WaitGPU();
        g_uploads.clear();

        HR(g_alloc[g_frame]->Reset());
        HR(g_cl->Reset(g_alloc[g_frame].Get(), nullptr));
        HR(g_cl->Close());

        InitD2D(); // D3D11On12 + D2D (must come after GPU idle)
    }
    catch (const std::exception& ex)
    {
        MessageBoxA(nullptr, ex.what(), "Initialization Error",
                    MB_OK | MB_ICONERROR);
        return 1;
    }

    ShowWindow(g_hwnd, nCmdShow);
    UpdateWindow(g_hwnd);

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            Update();
            Render();
        }
    }

    Cleanup();
    return static_cast<int>(msg.wParam);
}

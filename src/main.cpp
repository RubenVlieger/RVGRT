#define CONSOLE
#include <windows.h>
#include "State.hpp"	
#include "StateRender.cuh"
#include <windowsx.h>
#include <chrono>
#include "Timer.hpp"

using std::cout;
using std::cerr;
using std::endl;
const char g_szClassName[] = "myWindowClass";
HBITMAP hBitmap = NULL;
HGDIOBJ hOldBitmap = NULL;

char* bmp = NULL;

RECT windowRect;
bool fixMousePos = true;
// Step 4: the Window Procedure

auto timeOfPreviousFrame = std::chrono::high_resolution_clock::now();


extern "C" 
{
  __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

void WndCreate(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
#ifdef CONSOLE   
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
#endif

    HDC hdc = GetDC(hwnd);

    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof(BITMAPINFO));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = State::dispWIDTH;
    bmi.bmiHeader.biHeight = -State::dispHEIGHT; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    bmi.bmiHeader.biSizeImage = (unsigned int)(State::dispWIDTH * State::dispHEIGHT * 4);
    hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**) &bmp, NULL, NULL);
    ReleaseDC(NULL, hdc);

    for(int i = 0; i < State::dispWIDTH*State::dispHEIGHT; i++) ((unsigned int*)bmp)[i] = (0xFF00FFFFu);

    hOldBitmap = SelectObject(hdc, hBitmap);
    State::state.setBitMap(bmp);

    if (hBitmap == NULL) {
        DWORD lastError = GetLastError();
        cerr << "NO BITMAP WAS CREATED " << endl;
    }
    State::state.Create();
}



void WndDraw(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{    
    PAINTSTRUCT     ps;
    HDC             hdc;
    BITMAP          bitmap;
    HDC             hdcMem;


    State::state.character.Update();
    State::state.deltaTime = 0.016f;

    //Timer t("drawCUDA");
    State::state.render->drawCUDA(State::state.character.camera.pos,
                                  State::state.character.camera.forward,
                                  State::state.character.camera.up,
                                  State::state.character.camera.right);
    
    State::state.render->framebuffer.readback((uint32_t*)State::state.bmp);
    //t.s();

    State::state.deltaXMouse = 0;
    State::state.deltaYMouse = 0;

    hdc = BeginPaint(hWnd, &ps);

    hdcMem = CreateCompatibleDC(hdc);
    hOldBitmap = SelectObject(hdcMem, hBitmap);

    GetObject(hBitmap, sizeof(bitmap), &bitmap);
    BitBlt(hdc, 0, 0, bitmap.bmWidth, bitmap.bmHeight, hdcMem, 0, 0, SRCCOPY);

    SelectObject(hdcMem, hOldBitmap);
    DeleteDC(hdcMem);

    EndPaint(hWnd, &ps);


    char title[200]; 
    snprintf(title, sizeof(title), "Ruben leip programma: %.2f ms", (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - timeOfPreviousFrame).count())); // NEW
    SetWindowTextA(hWnd, title); // NEW

    timeOfPreviousFrame = std::chrono::high_resolution_clock::now();
}
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    //cout << fixMousePos << endl;
    switch(msg)
    {
        case WM_LBUTTONDOWN:
        break;
        case WM_RBUTTONDOWN:
        break;  
        
        case WM_KEYDOWN:
        {
            unsigned long KC = (unsigned long)wParam;

            State::state.keysPressed.set(KC, 1);

            if(KC == VK_ESCAPE) {
                DestroyWindow(hwnd);
                //fixMousePos = !fixMousePos;
            }
        }
        break;
        case WM_KEYUP:
        {
            unsigned char KC = (unsigned char)wParam;
            State::state.keysPressed.set(KC, 0);
        }
        break;

        case WM_CREATE:
            WndCreate(hwnd, msg, wParam, lParam);
        break;
        
        case WM_MOUSEMOVE:
        break;

        case WM_SIZE:
            GetWindowRect(hwnd, &windowRect);
        break;

        case WM_INPUT: 
        {
            constexpr unsigned int rawdatabuffersize = sizeof(RAWINPUT);
            static RAWINPUT raw[sizeof(RAWINPUT)];
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw, (unsigned int*)&rawdatabuffersize, sizeof(RAWINPUTHEADER));

            if (raw->header.dwType == RIM_TYPEMOUSE)
            {
                long mouseX = raw->data.mouse.lLastX;
                long mouseY = raw->data.mouse.lLastY;

                State::state.deltaXMouse += mouseX;
                State::state.deltaYMouse += mouseY;
            }
            if(fixMousePos) {
                while(ShowCursor(FALSE) >= 0);
                SetCursorPos(State::dispWIDTH / 2 + windowRect.left, State::dispHEIGHT / 2 + windowRect.top);
            }
            else
                ShowCursor(TRUE);
                //while(ShowCursor(TRUE) >= 0);
        }
        break;
 
        case WM_PAINT:
            WndDraw(hwnd, msg, wParam, lParam);
        break;

        case WM_CLOSE:
            DestroyWindow(hwnd);
        break;
        case WM_DESTROY:
            PostQuitMessage(0);
        break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    WNDCLASSEX wc;
    HWND hwnd;
    MSG Msg;

    //Step 1: Registering the Window Class
    wc.cbSize        = sizeof(WNDCLASSEX);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = WndProc;
    wc.cbClsExtra    = 0;
    wc.cbWndExtra    = 0;
    wc.hInstance     = hInstance;
    wc.hIcon         = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor       = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wc.lpszMenuName  = NULL;
    wc.lpszClassName = g_szClassName;
    wc.hIconSm       = LoadIcon(NULL, IDI_APPLICATION);


    if(!RegisterClassEx(&wc))
    {
        MessageBox(NULL, "Window Registration Failed!", "Error!",
            MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

    // Step 2: Creating the Window
    hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        g_szClassName,
        "Ruben leip programma",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, State::dispWIDTH, State::dispHEIGHT,
        NULL, NULL, hInstance, NULL);

    if(hwnd == NULL)
    {
        MessageBox(NULL, "Window Creation Failed!", "Error!",
            MB_ICONEXCLAMATION | MB_OK);
        return 0;
    }

	#ifndef HID_USAGE_PAGE_GENERIC
	#define HID_USAGE_PAGE_GENERIC ((unsigned short) 0x01)
	#endif
	#ifndef HID_USAGE_GENERIC_MOUSE
	#define HID_USAGE_GENERIC_MOUSE ((unsigned short) 0x02)
	#endif

    RAWINPUTDEVICE rid[1];
	rid[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
	rid[0].usUsage = HID_USAGE_GENERIC_MOUSE;
	rid[0].dwFlags = RIDEV_INPUTSINK;
	rid[0].hwndTarget = hwnd;
	RegisterRawInputDevices(rid, 1, sizeof(rid[0]));

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    GetWindowRect(hwnd, &windowRect);
    // Step 3: The Message Loop
    while(GetMessage(&Msg, NULL, 0, 0) > 0)
    {

        TranslateMessage(&Msg);
        DispatchMessage(&Msg);
        InvalidateRect(hwnd, NULL, FALSE);        
    
        //SleepEx(0.5, TRUE);
    }

    return (int)Msg.wParam;
}
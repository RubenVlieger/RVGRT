// Forward-declare the platform-specific entry points.
// The actual definitions are in their respective platform files.
#if defined(_WIN32)
    #include <windows.h>
    int WINAPI Win32Main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow);
#elif defined(__APPLE__)
    int MacOSMain(int argc, const char * argv[]);
#endif

// The one and only main function for the entire project.
int main(int argc, char *argv[])
{
#if defined(_WIN32)
    // On Windows, call the Win32 entry point.
    return Win32Main(GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOW);
#elif defined(__APPLE__)
    // On macOS, call the Cocoa entry point.
    return MacOSMain(argc, (const char**)argv);
#else
    // Handle other platforms like Linux if you add them later.
    #error "Unsupported platform!"
    return 1;
#endif
}
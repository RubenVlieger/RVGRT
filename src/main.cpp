// Forward-declare the platform-specific entry points.
// The actual definitions are in their respective platform files.
#if defined(_WIN32)
    #include <windows.h>
    int WINAPI Win32Main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow);
#elif defined(__APPLE__)
    int MacOSMain(int argc, const char * argv[]);
#endif

#if defined(_WIN32)
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // On Windows, call the Win32 entry point.
    return Win32Main(hInstance, hPrevInstance, lpCmdLine, nCmdShow);
}
#else
// The one and only main function for the entire project.
int main(int argc, char *argv[])
{
    // On macOS, call the Cocoa entry point.
    return MacOSMain(argc, (const char**)argv);
}
#endif
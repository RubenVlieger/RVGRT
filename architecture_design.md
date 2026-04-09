{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \'97\
## Architecture Design: In-Game Console / Chat System\
\
### Overview\
\
A Minecraft-style console overlay that integrates with the existing compute-based text rendering pipeline, the platform abstraction layer, and the multiplayer WebSocket protocol. The design is **fully platform-agnostic in logic**, with only event routing and GPU buffer management being platform-specific.\
\
---\
\
### 1. New Files\
\
```\
include/console/\
  GameConsole.hpp        \'97 Main coordinator (owns buffer, input, commands)\
  ConsoleBuffer.hpp      \'97 Message storage & scrolling\
  ConsoleInput.hpp       \'97 Text input state machine\
  CommandRegistry.hpp    \'97 Extensible command dispatch\
\
src/console/\
  GameConsole.cpp        \'97 Core logic (platform-agnostic)\
  ConsoleBuffer.cpp      \'97 Message ring buffer\
  ConsoleInput.cpp       \'97 Input accumulation & editing\
  CommandRegistry.cpp    \'97 Command registration & execution\
```\
\
### 2. Core Types\
\
```cpp\
// ConsoleBuffer.hpp\
enum class ConsoleMsgType : uint8_t \{\
    System,    // Welcome, /help output \'97 white\
    Command,   // Echo of executed command \'97 gray\
    Chat,      // Player chat messages \'97 yellow\
    Error      // Unknown command, etc. \'97 red\
\};\
\
struct ConsoleMessage \{\
    std::string text;          // The displayed text (may be wrapped)\
    ConsoleMsgType type;\
    float timestamp;           // Game time (for fade-out)\
    std::string senderName;   // "Server", player name, or empty for system\
\};\
```\
\
### 3. ConsoleBuffer\
\
```cpp\
class ConsoleBuffer \{\
public:\
    static constexpr size_t MAX_LINES = 200;\
\
    void AddMessage(const std::string& text, ConsoleMsgType type,\
                    const std::string& sender = "");\
\
    // Scrolling \'97 arrow keys\
    void ScrollUp(size_t lines = 1);\
    void ScrollDown(size_t lines = 1);\
    void ScrollToBottom();          // New message = auto-scroll\
\
    // For rendering \'97 returns up to `visibleCount` messages from bottom\
    // respecting the current scroll offset\
    std::vector<const ConsoleMessage*> GetVisibleLines(\
        size_t visibleCount = CONSOLE_VISIBLE_LINES) const;\
\
    size_t GetScrollOffset() const;\
    void ResetScrollOnNewMessage(); // Called internally on AddMessage\
\
private:\
    std::vector<ConsoleMessage> _messages;\
    size_t _scrollOffset = 0;  // 0 = bottom, increases as you scroll up\
\};\
```\
\
**Scrolling model**: `_scrollOffset` starts at 0 (bottom of history). `ScrollUp` increments it (seeing older messages), `ScrollDown` decrements it. When a new message arrives while at the bottom, offset stays 0 (auto-scroll). When scrolled up, new messages don't change offset (user is reading history).\
\
### 4. ConsoleInput\
\
```cpp\
class ConsoleInput \{\
public:\
    static constexpr size_t MAX_INPUT_LENGTH = 256;\
\
    void AppendChar(char c);\
    void Backspace();\
    void Delete();\
    void Clear();\
    void SetCursorPos(size_t pos);\
    void MoveCursorLeft();\
    void MoveCursorRight();\
\
    const std::string& GetText() const;\
    std::string GetDisplayText() const;  // Returns "> " + text + blinking cursor\
    size_t GetCursorPos() const;\
    bool IsEmpty() const;\
\
private:\
    std::string _input;\
    size_t _cursorPos = 0;\
\};\
```\
\
### 5. CommandRegistry\
\
```cpp\
using CommandFn = std::function<void(\
    const std::vector<std::string>& args,   // " /name foo bar " \uc0\u8594  args = \{"foo", "bar"\}\
    class GameConsole& console)>;            // Can add response messages\
\
struct CommandEntry \{\
    std::string name;\
    std::string description;   // Shown in /help\
    CommandFn handler;\
\};\
\
class CommandRegistry \{\
public:\
    void Register(const std::string& name,\
                  const std::string& description,\
                  CommandFn handler);\
\
    // Returns true if command was found and executed\
    bool Execute(const std::string& input, GameConsole& console);\
\
    // For /help \'97 lists name + description pairs\
    std::vector<std::pair<std::string, std::string>> GetAllCommands() const;\
\
private:\
    std::unordered_map<std::string, CommandEntry> _commands;\
\};\
```\
\
**Built-in commands (registered during `GameConsole::Initialize()`)**:\
\
| Command | Description |\
|---------|-------------|\
| `/help` | Lists all registered commands |\
| `/name <name>` | Sets your display name (default: "Player") |\
\
**Unknown command handling**: If input starts with `/` but no match, `Execute` returns false, and GameConsole adds: `Unknown command. Type /help to see all possible commands.`\
\
### 6. GameConsole (Coordinator)\
\
```cpp\
class GameConsole \{\
public:\
    GameConsole();\
\
    void Initialize();        // Register commands, add welcome message\
    void Open(char prefix = 0);  // 0 = no prefix, '/' = command prefix\
    void Close();\
    void Toggle();\
    bool IsOpen() const;\
\
    // Called from the game loop every frame\
    void Update(float deltaTime);\
\
    // Input events (routed from Platform when console is open)\
    void OnCharInput(char c);          // Printable character\
    void OnSpecialKey(SpecialKey key); // Enter, Backspace, Escape, ArrowUp, ArrowDown\
\
    // Network callbacks (called from PollUpdates thread, then marshalled to main)\
    void OnChatReceived(int clientId, const std::string& text);\
\
    // Rendering \'97 called from MetalRenderer::Draw() or equivalent\
    void Render(TextRenderer& textRenderer, uint32_t screenWidth, uint32_t screenHeight);\
\
    ConsoleBuffer& GetBuffer();\
    ConsoleInput& GetInput();\
    CommandRegistry& GetRegistry();\
\
    // Player identity (for chat messages)\
    void SetPlayerName(const std::string& name);\
    const std::string& GetPlayerName() const;\
\
private:\
    ConsoleBuffer _buffer;\
    ConsoleInput _input;\
    CommandRegistry _registry;\
\
    bool _isOpen = false;\
    char _openPrefix = 0;       // If opened with '/', prefill input\
    float _cursorBlinkTime = 0.0f;\
    bool _cursorVisible = true;\
    std::string _playerName = "Player";\
\
    void SubmitInput();         // Called on Enter key\
\};\
```\
\
**Key behaviors**:\
- `Open('/')` pre-fills the input with `/` and positions cursor after it\
- `Open(0)` or `Open('t')` opens with empty input\
- `SubmitInput()` checks if input starts with `/` \uc0\u8594  command lookup; else \u8594  send as chat message via NetworkClient\
- `Update()` handles cursor blink timer (500ms toggle)\
- `Render()` calls `textRenderer.AddText()` for the visible message lines + input line\
\
### 7. Rendering Integration\
\
The console is rendered in `MetalRenderer::Draw()` during Pass 8 (TextOverlay), using the existing `TextRenderer`:\
\
```cpp\
// In MetalRenderer::Draw(), replacing the current "Hello World" placeholder:\
if (_fontAtlas.IsValid()) \{\
    _textRenderer.BeginFrame(State::dispWIDTH, State::dispHEIGHT);\
\
    // Render console (handles both open and closed states)\
    State::state.console.Render(\
        _textRenderer, State::dispWIDTH, State::dispHEIGHT);\
\
    // Render player name tags (world-space) can still happen here\
    // ...\
\
    _textRenderer.EndFrame();\
    _textRenderer.UpdateBuffers((id<MTLDevice>)_device);\
\}\
```\
\
**GameConsole::Render() implementation outline**:\
\
```cpp\
void GameConsole::Render(TextRenderer& tr, uint32_t w, uint32_t h) \{\
    float lineH = CONSOLE_FONT_SIZE * CONSOLE_FONT_SCALE * 1.2f;\
\
    // Background rectangle when console is open \'97 rendered as a special\
    // "solid rect" glyph instance (see Section 8)\
    if (_isOpen) \{\
        // Draw semi-transparent dark background\
        tr.AddRect(0, h - CONSOLE_VISIBLE_LINES*lineH - lineH - 10,\
                   w, CONSOLE_VISIBLE_LINES*lineH + lineH + 20,\
                   simd_make_float4(0, 0, 0, 0.5f));\
    \}\
\
    // Messages \'97 always rendered (with fade when closed)\
    auto lines = _buffer.GetVisibleLines(CONSOLE_VISIBLE_LINES);\
    float alpha = _isOpen ? 0.95f : 0.7f;  // Slightly transparent like Minecraft\
\
    float y = h - lineH - 5.0f;  // Bottom of screen, going up\
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) \{\
        simd_float4 color = GetMsgColor((*it)->type, alpha);\
        tr.AddText((*it)->text, CONSOLE_MARGIN_X, y,\
                   CONSOLE_FONT_SCALE, color, 0.05f);\
        y -= lineH;\
    \}\
\
    // Input line (only when open)\
    if (_isOpen) \{\
        std::string display = "> " + _input.GetText();\
        if (_cursorVisible) display += "|";\
        tr.AddText(display, CONSOLE_MARGIN_X,\
                   h - CONSOLE_MARGIN_BOTTOM,\
                   CONSOLE_FONT_SCALE,\
                   simd_make_float4(1, 1, 1, 1.0f), 0.05f);\
    \}\
\}\
```\
\
**Message color mapping**:\
```cpp\
simd_float4 GetMsgColor(ConsoleMsgType type, float alpha) \{\
    switch (type) \{\
        case ConsoleMsgType::System:  return \{1.0, 1.0, 1.0, alpha\};    // White\
        case ConsoleMsgType::Command: return \{0.7, 0.7, 0.7, alpha\};    // Gray\
        case ConsoleMsgType::Chat:    return \{1.0, 1.0, 0.5, alpha\};    // Yellow\
        case ConsoleMsgType::Error:   return \{1.0, 0.3, 0.3, alpha\};    // Red\
    \}\
\}\
```\
\
### 8. TextRenderer Extension: Solid Rectangles\
\
To render the semi-transparent console background, we extend `GlyphInstance` with a second flag bit:\
\
```cpp\
// In ShaderTypes.h, modify flags semantics:\
// Bit 0: depth test enable (existing)\
// Bit 1: solid rectangle mode (new)\
// When flags & 2, screenPos/screenSize define a solid-color rect,\
// atlasUVMin/Max are ignored, softness is ignored\
```\
\
This requires a minor change in `text_overlay.shader`:\
\
```glsl\
// Inside the glyph loop, after AABB bounds check:\
if ((g.flags & 2u) != 0) \{\
    // Solid rectangle: fill the entire rect with g.color\
    float srcAlpha = g.color.w;\
    color = make_float4(\
        color.x * (1.0 - srcAlpha) + g.color.x * srcAlpha,\
        color.y * (1.0 - srcAlpha) + g.color.y * srcAlpha,\
        color.z * (1.0 - srcAlpha) + g.color.z * srcAlpha,\
        1.0\
    );\
    continue;  // Skip SDF sampling\
\}\
```\
\
**This is the only shader change needed**. Going through the tile system ensures we still get the culling benefit \'97 the background rect only affects tiles it overlaps.\
\
New method on TextRenderer:\
```cpp\
void AddRect(float x, float y, float w, float h,\
             simd_float4 color, bool depthTest = false, float sceneDepth = 1e30f);\
```\
\
This creates a `GlyphInstance` with `flags |= 2`, `atlasUVMin/Max` set to `(0,0)-(1,1)` (irrelevant), and the rect dimensions as `screenPos/screenSize`.\
\
### 9. Platform Input Changes\
\
**Platform.hpp \'97 additions**:\
\
```cpp\
// Console input support\
std::queue<char> textInputQueue;         // Printable characters (thread-safe from event thread)\
std::mutex textInputMutex;                // Protects the queue\
std::atomic<bool> consoleOpen\{false\};     // When true, game key/mouse input is suppressed\
```\
\
**GameView.mm \'97 changes**:\
\
In `keyDown:`:\
```objc\
- (void)keyDown:(NSEvent *)event \{\
    Platform* platform = State::state.platform.get();\
    if (!platform) return;\
\
    // If console is open, route all key events to console\
    if (State::state.console.IsOpen()) \{\
        if ([event isARepeat]) return;\
\
        NSString* chars = [event characters]; // Get the typed character\
        if (chars.length > 0) \{\
            char c = [chars characterAtIndex:0];\
            // Filter to printable ASCII (32-126) + handle special keys\
            if (c >= 32 && c < 127) \{\
                std::lock_guard<std::mutex> lock(platform->textInputMutex);\
                platform->textInputQueue.push(c);\
            \}\
        \}\
\
        // Special keys handled by key code\
        switch (event.keyCode) \{\
            case kVK_Return:  // Enter\
            case kVK_Delete:   // Backspace\
            case kVK_UpArrow:\
            case kVK_DownArrow:\
            case kVK_Escape:\
                platform->keysPressed.set(event.keyCode, 1);\
                break;\
        \}\
        return;  // Don't process as game input\
    \}\
\
    // Console toggle keys\
    if (event.keyCode == kVK_ANSI_T || event.keyCode == kVK_ANSI_Slash) \{\
        // 'T' opens chat, '/' opens command\
        if (!State::state.console.IsOpen()) \{\
            char prefix = (event.keyCode == kVK_ANSI_Slash) ? '/' : 0;\
            State::state.console.Open(prefix);\
            platform->consoleOpen = true;\
            [self setMouseLock:NO];  // Unlock mouse\
            return;\
        \}\
    \}\
\
    // Normal game input (existing code)\
    platform->keysPressed.set(event.keyCode, 1);\
    if (event.keyCode == kVK_Escape) \{\
        [self setMouseLock:!_mouseIsLocked];\
    \}\
\}\
```\
\
**Key routing in the game loop** (`macos_main.mm` or `Character::Update()`):\
\
```cpp\
// In gameLoop: or Character::Update():\
if (State::state.console.IsOpen()) \{\
    // Drain text input queue\
    auto& platform = State::state.platform;\
    std::lock_guard<std::mutex> lock(platform->textInputMutex);\
    while (!platform->textInputQueue.empty()) \{\
        char c = platform->textInputQueue.front();\
        platform->textInputQueue.pop();\
        State::state.console.OnCharInput(c);\
    \}\
\
    // Check special keys\
    if (platform->IsKeyDown(/*kVK_Return*/)) \{\
        State::state.console.OnSpecialKey(SpecialKey::Enter);\
        if (!State::state.console.IsOpen()) \{\
            platform->consoleOpen = false;\
            // Re-lock mouse\
        \}\
    \}\
    if (platform->IsKeyDown(/*kVK_Delete*/)) \{\
        State::state.console.OnSpecialKey(SpecialKey::Backspace);\
    \}\
    // ... arrow keys, escape\
\
    // Skip character movement when console is open\
    return;\
\}\
```\
\
**Escape key** closes the console. **Enter key** submits input. **Backspace** deletes. **Arrow Up/Down** scrolls history. When the console closes, mouse re-locks and game input resumes.\
\
### 10. NetworkClient Extension\
\
**NetworkClient.hpp \'97 additions**:\
\
```cpp\
// Chat callback type\
using ChatCallback = std::function<void(int clientId, const std::string& text)>;\
\
class NetworkClient \{\
    // Existing methods...\
    virtual void SendChat(const std::string& text) = 0;  // NEW\
    void SetChatCallback(ChatCallback cb);                // NEW\
protected:\
    ChatCallback _chatCallback;\
\};\
```\
\
**MacOSNetworkClient.mm \'97 changes in ReadLoop()**:\
\
```objc\
// Already handles "chat" type - add callback dispatch:\
if ([type isEqualToString:@"chat"]) \{\
    int senderId = [json[@"client_id"] intValue];\
    NSString* text = json[@"text"];\
    if (weakThis->_chatCallback) \{\
        weakThis->_chatCallback(senderId, [text UTF8String]);\
    \}\
\}\
```\
\
**MacOSNetworkClient::SendChat()** \'97 new method:\
\
```objc\
- (void)SendChat:(const std::string&)text \{\
    if (!webSocketTask || webSocketTask.state != NSURLSessionTaskStateRunning) return;\
    \
    NSDictionary* payload = @\{\
        @"type": @"chat",\
        @"text": [NSString stringWithUTF8String:text.c_str()]\
    \};\
    // Serialize and send like SendState does\
    // ...\
\}\
```\
\
**Connection in macos_main.mm** \'97 wire up the callback:\
\
```cpp\
State::state.networkClient->SetChatCallback([](int clientId, const std::string& text) \{\
    State::state.console.OnChatReceived(clientId, text);\
\});\
```\
\
### 11. State Integration\
\
**State.hpp \'97 addition**:\
\
```cpp\
#include "console/GameConsole.hpp"\
\
class State \{\
public:\
    // Existing members...\
    GameConsole console;  // The in-game console\
    // ...\
\};\
```\
\
### 12. SystemConfig.h Additions\
\
```cpp\
// Console settings\
#define CONSOLE_MAX_LINES 200\
#define CONSOLE_VISIBLE_LINES 20\
#define CONSOLE_INPUT_MAX_LENGTH 256\
#define CONSOLE_FADE_TIME 5.0f        // Seconds before closed messages fully fade\
#define CONSOLE_LINE_HEIGHT 22.0f     // Pixels between lines\
#define CONSOLE_FONT_SCALE 0.9f       // Text scale for console\
#define CONSOLE_MARGIN_X 12.0f\
#define CONSOLE_MARGIN_BOTTOM 30.0f    // Input line offset from bottom\
#define CONSOLE_BG_ALPHA 0.5f          // Background rectangle opacity\
#define CONSOLE_TEXT_ALPHA 0.9f        // Text opacity when open\
#define CONSOLE_TEXT_ALPHA_FADED 0.4f  // Text opacity when closed/fading\
#define CURSOR_BLINK_INTERVAL 0.5f     // Seconds per cursor blink cycle\
```\
\
### 13. Cross-Platform Considerations\
\
| Concern | macOS (Metal) | Windows (CUDA/D3D12) | Future (WebGPU) |\
|---------|---------------|----------------------|-----------------|\
| **Console logic** | Shared C++ \'97 `GameConsole.cpp` | Same | Same |\
| **Input routing** | `GameView.mm` \uc0\u8594  `Platform::textInputQueue` | `Win32Platform.cpp` \u8594  same queue | Emscripten keyboard events \u8594  same queue |\
| **Text rendering** | `TextRenderer.mm` + `.shader` | `TextRenderer.cpp` + CUDA/D3D12 shader | `TextRenderer.cpp` + WGSL shader |\
| **Solid rect fallback** | `text_overlay.shader` flag bit 1 | Same `.shader` \uc0\u8594  CUDA transpiler | WGSL port |\
| **Network** | `MacOSNetworkClient.mm` (NSURLSession WebSocket) | `Win32NetworkClient.cpp` (WinHTTP WebSocket) | Emscripten WebSocket API |\
\
**The `.shader` file** is already designed to transpile across Metal, CUDA, and potentially WGSL. The new `flags & 2` solid-rect check follows the existing preprocessor pattern and will work on all backends without branching.\
\
### 14. File Dependency Graph\
\
```\
macos_main.mm (gameLoop)\
  \uc0\u9500 \u9472 \u9472  GameConsole::Update(dt)           \'97 cursor blink, scroll\
  \uc0\u9500 \u9472 \u9472  GameConsole::OnCharInput()        \'97 process text queue\
  \uc0\u9500 \u9472 \u9472  GameConsole::OnSpecialKey()       \'97 enter/special keys\
  \uc0\u9474 \
  \uc0\u9500 \u9472 \u9472  Character::Update()               \'97 SKIPPED when console is open\
  \uc0\u9474 \
  \uc0\u9492 \u9472 \u9472  MetalRenderer::Draw()\
        \uc0\u9492 \u9472 \u9472  GameConsole::Render(textRenderer, w, h)\
              \uc0\u9500 \u9472 \u9472  TextRenderer::AddRect()          \'97 background\
              \uc0\u9492 \u9472 \u9472  TextRenderer::AddText()           \'97 messages + input line\
                    \uc0\u9492 \u9472 \u9472  TextOverlay (compute shader) \'97 GPU rendering\
\
NetworkClient\
  \uc0\u9492 \u9472 \u9472  SetChatCallback \u8594  GameConsole::OnChatReceived()\
  \uc0\u9492 \u9472 \u9472  GameConsole::SubmitInput() \u8594  NetworkClient::SendChat()\
```\
\
### 15. Implementation Order\
\
1. **`ConsoleBuffer` + `ConsoleInput` + `CommandRegistry`** \'97 Pure data classes, no dependencies, trivially testable.\
2. **`GameConsole`** \'97 Ties the above together. Unit testable with mock buffer.\
3. **TextRenderer extension** \'97 Add `AddRect()` + `GlyphInstance` flag bit 2 + shader change.\
4. **`Platform.hpp` changes** \'97 Add `textInputQueue`, `consoleOpen`, `textInputMutex`.\
5. **`GameView.mm` key routing** \'97 Intercept `T`/`/` to open console, route text when open.\
6. **`macos_main.mm` game loop** \'97 Process console input, suppress character movement, wire chat callback.\
7. **`NetworkClient` extension** \'97 Add `SendChat()` + `ChatCallback`.\
8. **`MetalRenderer::Draw()`** \'97 Replace "Hello World" with `GameConsole::Render()`.\
9. **`State.hpp`** \'97 Add `GameConsole console` member.\
\
This order ensures each step is independently testable: you can verify console logic without GPU changes, test text rendering with rects without input handling, etc.\
\
### 16. Key Design Decisions Summary\
\
| Decision | Rationale |\
|----------|-----------|\
| **Solid-rect via GlyphInstance flag** instead of separate shader pass | Minimal shader change, reuses tile culling, zero extra draw calls |\
| **Platform-agnostic console logic** in C++ | Share between macOS/CUDA/WebGPU \'97 only event routing differs |\
| **`textInputQueue` + mutex** on Platform | Clean thread boundary: event thread pushes, game loop drains |\
| **Command registry pattern** | New commands = one `Register()` call, no switch statements, easy to extend |\
| **ConsoleMsgType enum** | Colors drive readability; yellow chat, white system, red errors, gray commands |\
| **Console lives in State** | Follows the project's singleton pattern; accessible from renderer, input, network |\
| **Open('/') vs Open(0)** | Minecraft-style: `/` opens with command prefix, `T` opens blank |\
| **Auto-scroll with manual override** | Scroll offset resets to 0 on new messages if already at bottom; user can scroll up to read history |}
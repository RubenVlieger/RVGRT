#include "console/RegisterCommands.hpp"
#include "console/GameConsole.hpp"
#include "State.hpp"
#include "VoxelQuery.hpp"
#include <sstream>
#include <cstdlib>
//
// This file registers all in-game console commands. It is the single
// canonical location for command definitions — DO NOT spread command
// registration across multiple files.
//
// Architecture
// ─────────────
// Each command is a static function with the signature:
//   void cmd_name(const std::vector<std::string>& args, GameConsole& console)
// The args vector contains the space-separated tokens after the command name
// (e.g. "/speed 0.1" → args = {"0.1"}). The console reference lets the handler
// print output via GetBuffer().AddMessage(...).
//
// Commands are registered by calling reg.Register("name", "description", cmd_fn).
// Registration order is alphabetical within each category block.
//
// Adding a New Command
// ─────────────────────
// 1. Write a static handler function (see examples below).
// 2. Add one reg.Register(...) call in RegisterAllCommands().
// That's it — the CommandRegistry handles parsing, and SubmitInput() handles
// the /prefix, echo, and error reporting automatically.
//
// ConsoleMsgType Colors
// ─────────────────────
//   System  — white  — informational messages (spawn confirmation, etc.)
//   Command — gray   — echoed command lines
//   Chat    — yellow — player chat / emotes
//   Error   — red    — invalid usage, unknown commands
//
// State Access
// ─────────────
// Handlers access game state via State::state (the global singleton).
// Key mutable fields:
//   State::state.character     — player position, speed, FOV, sensitivity, etc.
//   State::state.homePosition   — saved home for /sethome + /home
//   State::state.flyMode        — stub for future fly mode
//   State::state.fpsInfo       — updated each frame by the render loop
//   State::state.otherCharacters — visible network players
//
// See Also
// ─────────
//   include/console/GameConsole.hpp  — console coordinator
//   include/console/CommandRegistry.hpp — registration + dispatch logic
//   include/console/ConsoleBuffer.hpp  — message buffer + scrolling
//   include/console/ConsoleInput.hpp   — text input + history
//   include/State.hpp                 — global state (homePosition, flyMode, etc.)
// ============================================================================

#include "console/RegisterCommands.hpp"
#include "console/GameConsole.hpp"
#include "State.hpp"
#include "VoxelQuery.hpp"
#include <sstream>
#include <cstdlib>

// ============================================================================
// Helper: print a formatted message to the console
// ============================================================================
static void Print(GameConsole& console, const std::string& text,
                  ConsoleMsgType type = ConsoleMsgType::System) {
    console.GetBuffer().AddMessage(text, type);
}

static void PrintF(GameConsole& console, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void PrintF(GameConsole& console, const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    console.GetBuffer().AddMessage(buf, ConsoleMsgType::System);
}

// ============================================================================
// Essential Commands (/help and /name — must be registered before others)
// ============================================================================
static void cmd_help(const std::vector<std::string>&, GameConsole& console) {
    auto commands = console.GetRegistry().GetAllCommands();
    console.GetBuffer().AddMessage("--- Available commands ---",
                                   ConsoleMsgType::System);
    for (const auto& [name, desc] : commands) {
        console.GetBuffer().AddMessage("/" + name + " - " + desc,
                                       ConsoleMsgType::Command);
    }
}

static void cmd_name(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        console.GetBuffer().AddMessage("Usage: /name <your_name>",
                                       ConsoleMsgType::Error);
        return;
    }
    std::string newName;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) newName += " ";
        newName += args[i];
    }
    console.SetPlayerName(newName);
    console.GetBuffer().AddMessage("Name set to: " + newName,
                                   ConsoleMsgType::System);
}

// ============================================================================
// Movement & Navigation Commands
// ============================================================================
static void cmd_spawn(const std::vector<std::string>&, GameConsole& console) {
    State::state.character.position = glm::vec3(508.0f, 156.0f, 408.0f);
    State::state.character.velocity = glm::vec3(0.0f);
    Print(console, "Teleported to world spawn.");
}

static void cmd_home(const std::vector<std::string>&, GameConsole& console) {
    State::state.character.position = State::state.homePosition;
    State::state.character.velocity = glm::vec3(0.0f);
    PrintF(console, "Teleported to home at (%.1f, %.1f, %.1f).",
           State::state.homePosition.x, State::state.homePosition.y,
           State::state.homePosition.z);
}

static void cmd_sethome(const std::vector<std::string>&, GameConsole& console) {
    State::state.homePosition = State::state.character.position;
    PrintF(console, "Home set to (%.1f, %.1f, %.1f).",
           State::state.homePosition.x, State::state.homePosition.y,
           State::state.homePosition.z);
}

static void cmd_tp(const std::vector<std::string>& args, GameConsole& console) {
    if (args.size() < 3) {
        Print(console, "Usage: /tp <x> <y> <z>", ConsoleMsgType::Error);
        return;
    }
    float x = std::atof(args[0].c_str());
    float y = std::atof(args[1].c_str());
    float z = std::atof(args[2].c_str());
    State::state.character.position = glm::vec3(x, y, z);
    State::state.character.velocity = glm::vec3(0.0f);
    PrintF(console, "Teleported to (%.1f, %.1f, %.1f).", x, y, z);
}

static void cmd_jump(const std::vector<std::string>& args, GameConsole& console) {
    if (args.size() < 3) {
        Print(console, "Usage: /jump <dx> <dy> <dz>", ConsoleMsgType::Error);
        return;
    }
    float dx = std::atof(args[0].c_str());
    float dy = std::atof(args[1].c_str());
    float dz = std::atof(args[2].c_str());
    State::state.character.position += glm::vec3(dx, dy, dz);
    PrintF(console, "Jumped by (%.1f, %.1f, %.1f) to (%.1f, %.1f, %.1f).",
           dx, dy, dz,
           State::state.character.position.x,
           State::state.character.position.y,
           State::state.character.position.z);
}

// ============================================================================
// Player Settings Commands
// ============================================================================
static void cmd_speed(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        PrintF(console, "Current speed: %.3f", State::state.character.speed);
        return;
    }
    float val = std::atof(args[0].c_str());
    if (val <= 0.0f) {
        Print(console, "Speed must be positive.", ConsoleMsgType::Error);
        return;
    }
    State::state.character.speed = val;
    PrintF(console, "Speed set to %.3f.", val);
}

static void cmd_gravity(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        PrintF(console, "Current gravity: %.3f", State::state.character.gravityAmount);
        return;
    }
    float val = std::atof(args[0].c_str());
    State::state.character.gravityAmount = val;
    PrintF(console, "Gravity set to %.3f.", val);
}

static void cmd_fly(const std::vector<std::string>&, GameConsole& console) {
    Print(console, "Flying is not yet implemented.", ConsoleMsgType::System);
}

static void cmd_noclip(const std::vector<std::string>&, GameConsole& console) {
    bool newMode = !State::state.noclipMode;
    State::state.noclipMode = newMode;

    if (newMode) {
        State::state.character.gravityAmount = 0.0f;
        State::state.character.onGround = false;
        State::state.character.velocity = glm::vec3(0.0f);
        Print(console, "Noclip enabled — free flight active.", ConsoleMsgType::System);
    } else {
        State::state.character.gravityAmount = -24.0f;
        State::state.character.jumpSpeed = 7.5f;
        State::state.character.velocity.x = 0.0f;
        State::state.character.velocity.z = 0.0f;
        State::state.character.velocity.y = 0.0f;
        State::state.character.onGround = false;

        // Search downward for ground surface and snap player onto it
        glm::vec3 pos = State::state.character.position;
        float feetY = pos.y - State::state.character.playerHeight;
        int startBlockY = int(floor(feetY));
        int blockX = int(floor(pos.x));
        int blockZ = int(floor(pos.z));

        bool foundGround = false;
        for (int y = startBlockY; y > startBlockY - 256; --y) {
            if (IsVoxelSolid(blockX, y, blockZ)) {
                State::state.character.position.y = float(y + 1) + State::state.character.playerHeight;
                State::state.character.onGround = true;
                foundGround = true;
                break;
            }
        }

        if (!foundGround) {
            Print(console, "Noclip disabled — no ground found below! You will fall.", ConsoleMsgType::Error);
        } else {
            Print(console, "Noclip disabled — collision enabled.", ConsoleMsgType::System);
        }
    }
}

static void cmd_fov(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        PrintF(console, "Current FOV: %.1f degrees.", State::state.character.FOV);
        return;
    }
    float val = std::atof(args[0].c_str());
    if (val < 10.0f || val > 170.0f) {
        Print(console, "FOV must be between 10 and 170.", ConsoleMsgType::Error);
        return;
    }
    State::state.character.FOV = val;
    PrintF(console, "FOV set to %.1f degrees.", val);
}

static void cmd_sensitivity(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        PrintF(console, "Current sensitivity: %.5f", State::state.character.sensitivity);
        return;
    }
    float val = std::atof(args[0].c_str());
    if (val <= 0.0f) {
        Print(console, "Sensitivity must be positive.", ConsoleMsgType::Error);
        return;
    }
    State::state.character.sensitivity = val;
    PrintF(console, "Sensitivity set to %.5f.", val);
}

static void cmd_reset(const std::vector<std::string>&, GameConsole& console) {
    auto& c = State::state.character;
    c.speed = 0.05f;
    c.sensitivity = 0.00003f;
    c.gravityAmount = 0.0f;
    c.FOV = 70.0f;
    c.jumpSpeed = 2.0f;
    c.onGround = false;
    State::state.flyMode = false;
    State::state.noclipMode = true;

    // Signal that block changes should be reset (handled in game loop)
    State::state.blockResetRequested = true;
    State::state.localBlockEdits.clear();

    // Notify the server so it resets its authority store and broadcasts
    // the reset to all other clients.
    auto cb = console.GetChatSendCallback();
    if (cb) {
        cb(console.GetPlayerName(), "/reset");
    }

    Print(console, "All settings and block changes reset to defaults.");
}

// ============================================================================
// Informational Commands
// ============================================================================
static void cmd_pos(const std::vector<std::string>&, GameConsole& console) {
    const auto& c = State::state.character;
    PrintF(console, "Position: (%.2f, %.2f, %.2f)", c.position.x, c.position.y, c.position.z);
    PrintF(console, "Direction: (%.3f, %.3f, %.3f)", c.direction.x, c.direction.y, c.direction.z);
    PrintF(console, "Yaw: %.2f  Pitch: %.2f", c.yaw, c.pitch);
}

static void cmd_fps(const std::vector<std::string>&, GameConsole& console) {
    console.GetBuffer().AddMessage(State::state.fpsInfo, ConsoleMsgType::System);
}

static void cmd_players(const std::vector<std::string>&, GameConsole& console) {
    PrintF(console, "You: %s (client %d)",
           State::state.console.GetPlayerName().c_str(),
           State::state.console.GetPlayerClientId());
    PrintF(console, "Other players visible: %zu",
           State::state.otherCharacters.size());
}

static void cmd_me(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        Print(console, "Usage: /me <message>", ConsoleMsgType::Error);
        return;
    }
    std::string msg;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) msg += " ";
        msg += args[i];
    }
    std::string emote = "* " + State::state.console.GetPlayerName() + " " + msg + " *";
    console.GetBuffer().AddMessage(emote, ConsoleMsgType::Chat);
    auto cb = console.GetChatSendCallback();
    if (cb) {
        cb(State::state.console.GetPlayerName(), emote);
    }
}

static void cmd_time(const std::vector<std::string>&, GameConsole& console) {
    Print(console, "Time of day control is not yet implemented.", ConsoleMsgType::System);
}

static void cmd_clear(const std::vector<std::string>&, GameConsole& console) {
    console.GetBuffer().Clear();
}

static void cmd_mat(const std::vector<std::string>& args, GameConsole& console) {
    if (args.empty()) {
        PrintF(console, "Current material: %d (use /mat <id> to change)", State::state.selectedMaterialID);
        Print(console, "Common IDs: 1=stone 2=grass 3=dirt 4=cobble 5=planks 7=bedrock", ConsoleMsgType::System);
        return;
    }
    int val = std::atoi(args[0].c_str());
    if (val < 0 || val > 255) {
        Print(console, "Material ID must be 0-255.", ConsoleMsgType::Error);
        return;
    }
    State::state.selectedMaterialID = static_cast<uint8_t>(val);
    PrintF(console, "Placement material set to %d.", val);
}

// ============================================================================
// Register All Commands
// ============================================================================
void RegisterAllCommands(GameConsole& console) {
    auto& reg = console.GetRegistry();

    // --- Essential ---
    reg.Register("help",    "Show all available commands",             cmd_help);
    reg.Register("name",    "Set your display name",                  cmd_name);

    // --- Navigation ---
    reg.Register("spawn",    "Teleport to world spawn point",          cmd_spawn);
    reg.Register("home",    "Teleport to your saved home point",      cmd_home);
    reg.Register("sethome", "Save your current position as home",     cmd_sethome);
    reg.Register("tp",       "Teleport to absolute coordinates",      cmd_tp);
    reg.Register("jump",    "Jump by a relative offset",              cmd_jump);

    // --- Player Settings ---
    reg.Register("speed",      "Set movement speed (default 0.05)",       cmd_speed);
    reg.Register("gravity",    "Set gravity amount (default 0.0)",        cmd_gravity);
    reg.Register("fly",        "Toggle fly mode",                         cmd_fly);
    reg.Register("noclip",     "Toggle noclip mode",                      cmd_noclip);
    reg.Register("fov",        "Set field of view in degrees (default 70)", cmd_fov);
    reg.Register("sensitivity","Set mouse sensitivity (default 0.00003)", cmd_sensitivity);
    reg.Register("reset",      "Reset all settings to defaults",          cmd_reset);

    // --- Information ---
    reg.Register("pos",     "Show current position and direction",    cmd_pos);
    reg.Register("fps",     "Show frame timing information",        cmd_fps);
    reg.Register("players", "Show connected players",                cmd_players);
    reg.Register("me",      "Send an emote message",                cmd_me);
    reg.Register("time",    "Set sun direction for time of day",    cmd_time);
    reg.Register("clear",   "Clear all console messages",            cmd_clear);
    reg.Register("mat",     "Set placement material ID (default 2=grass)", cmd_mat);
}

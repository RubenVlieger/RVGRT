#pragma once

// Forward declaration
class GameConsole;

// Registers all built-in console commands with the given GameConsole instance.
// Call this once from GameConsole::Initialize() after the registry is constructed.
void RegisterAllCommands(GameConsole& console);

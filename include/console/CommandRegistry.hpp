#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration — avoids circular include
class GameConsole;

// Command handler function: receives parsed arguments and a reference
// to the console so it can add response messages.
using CommandFn = std::function<void(
    const std::vector<std::string>& args,
    GameConsole& console)>;

struct CommandEntry {
    std::string name;
    std::string description;
    CommandFn handler;
};

// Extensible command registry with argument parsing.
// Commands are registered with a name, description, and handler.
// Execute() parses "/name arg1 arg2 ..." and dispatches to the handler.
class CommandRegistry {
public:
    CommandRegistry() = default;
    ~CommandRegistry() = default;

    // Register a command. Overwrites any existing command with the same name.
    void Register(const std::string& name,
                  const std::string& description,
                  CommandFn handler);

    // Parse and execute a command line (e.g. "/help" or "/name John").
    // Returns true if the command was found and executed.
    // Returns false if the command was not found (caller should show an error).
    // The input MUST start with '/'.
    bool Execute(const std::string& input, GameConsole& console);

    // Returns all registered commands sorted alphabetically (for /help)
    std::vector<std::pair<std::string, std::string>> GetAllCommands() const;

    bool HasCommand(const std::string& name) const;

private:
    std::unordered_map<std::string, CommandEntry> _commands;

    struct ParsedCommand {
        std::string name;
        std::vector<std::string> args;
    };

    ParsedCommand ParseInput(const std::string& input);
};
#include "console/CommandRegistry.hpp"
#include "console/GameConsole.hpp"
#include <algorithm>
#include <sstream>

void CommandRegistry::Register(const std::string& name,
                               const std::string& description,
                               CommandFn handler) {
    CommandEntry entry;
    entry.name = name;
    entry.description = description;
    entry.handler = std::move(handler);
    _commands[name] = std::move(entry);
}

bool CommandRegistry::Execute(const std::string& input, GameConsole& console) {
    if (input.empty() || input[0] != '/') {
        return false;
    }

    auto parsed = ParseInput(input);

    auto it = _commands.find(parsed.name);
    if (it == _commands.end()) {
        return false;
    }

    it->second.handler(parsed.args, console);
    return true;
}

std::vector<std::pair<std::string, std::string>> CommandRegistry::GetAllCommands() const {
    std::vector<std::pair<std::string, std::string>> result;
    result.reserve(_commands.size());

    for (const auto& [name, entry] : _commands) {
        result.emplace_back(entry.name, entry.description);
    }

    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    return result;
}

bool CommandRegistry::HasCommand(const std::string& name) const {
    return _commands.find(name) != _commands.end();
}

CommandRegistry::ParsedCommand CommandRegistry::ParseInput(const std::string& input) {
    ParsedCommand result;

    // Skip the leading '/'
    std::string cmdLine = (input[0] == '/') ? input.substr(1) : input;

    std::istringstream iss(cmdLine);
    std::string token;

    // First token is the command name
    if (std::getline(iss, token, ' ')) {
        result.name = token;
    }

    // Remaining tokens are arguments (each space-separated word)
    while (std::getline(iss, token, ' ')) {
        if (!token.empty()) {
            result.args.push_back(token);
        }
    }

    return result;
}
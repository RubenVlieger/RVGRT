#pragma once
#include <memory>
#include <vector>
#include <string>

// Forward declaration
class Character;

class NetworkClient {
public:
    virtual ~NetworkClient() = default;

    // Connects asynchronously to the specified WebSocket URL (e.g., "ws://127.0.0.1:8000/ws")
    virtual void Connect(const std::string& url) = 0;

    // Disconnects from the server
    virtual void Disconnect() = 0;

    // Serializes the local character's matrices to JSON and sends them
    virtual void SendState(const Character& localCharacter) = 0;

    // Called once per frame on the main thread.
    // Drains any newly received states and directly updates the destination array
    // (State::state.otherCharacters).
    virtual void PollUpdates(std::vector<Character>& otherCharacters) = 0;

    // Factory method to allocate the OS-specific implementation
    static std::unique_ptr<NetworkClient> Create();
};

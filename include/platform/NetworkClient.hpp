#pragma once
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cstdint>

// Forward declaration
class Character;

#include "BlockInteraction.hpp"

// Callback type for incoming chat messages: (clientId, senderName, text)
using ChatCallback = std::function<void(int clientId, const std::string& sender, const std::string& text)>;

// Callback type for incoming block edits from other players
using BlockEditCallback = std::function<void(int32_t x, int32_t y, int32_t z, uint8_t matID)>;

// Callback type for full block sync on connect (receives all accumulated changes)
using BlockSyncCallback = std::function<void(const std::vector<BlockEdit>& edits)>;

// Callback type for block reset notification from server
using BlockResetCallback = std::function<void()>;

class NetworkClient {
public:
    virtual ~NetworkClient() = default;

    // Connects asynchronously to the specified WebSocket URL (e.g., "ws://127.0.0.1:8000/ws")
    virtual void Connect(const std::string& url) = 0;

    // Disconnects from the server
    virtual void Disconnect() = 0;

    // Serializes the local character's matrices to JSON and sends them
    virtual void SendState(const Character& localCharacter) = 0;

    // Sends a chat message to the server
    virtual void SendChat(const std::string& sender, const std::string& text) = 0;

    // Sends a block edit to the server
    virtual void SendBlockEdit(int32_t x, int32_t y, int32_t z, uint8_t matID) = 0;

    // Called once per frame on the main thread.
    // Drains any newly received states and directly updates the destination array
    // (State::state.otherCharacters).
    virtual void PollUpdates(std::vector<Character>& otherCharacters) = 0;

    // Set the callback for incoming chat messages
    virtual void SetChatCallback(ChatCallback callback) = 0;

    // Set the callback for incoming block edits from other players
    virtual void SetBlockEditCallback(BlockEditCallback callback) = 0;

    // Set the callback for full block sync on connect
    virtual void SetBlockSyncCallback(BlockSyncCallback callback) = 0;

    // Set the callback for block reset notification from server
    virtual void SetBlockResetCallback(BlockResetCallback callback) = 0;

    // Factory method to allocate the OS-specific implementation
    static std::unique_ptr<NetworkClient> Create();
};

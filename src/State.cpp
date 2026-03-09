#include "State.hpp"
#include "platform/Platform.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/Renderer.hpp"
#include "platform/NetworkClient.hpp"

// The global singleton instance
State State::state = State();

int State::dispWIDTH = 1920;
int State::dispHEIGHT = 1080;
int State::screenWIDTH = 1920;
int State::screenHEIGHT = 1080;

State::State()
{
    // The constructor can be empty if initialization is done in the main function.
    Character npc;
    otherCharacters.push_back(npc);
}

// DEFINE the destructor here. 
// Now, the compiler has the full definitions of the interfaces and can correctly destroy the unique_ptrs.
State::~State() = default;
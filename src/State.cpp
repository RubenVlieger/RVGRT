#include "State.hpp"
#include "platform/Platform.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/Renderer.hpp"

// The global singleton instance
State State::state = State();

int State::dispWIDTH = 3072 * 0.6f;
int State::dispHEIGHT = 1920 * 0.6f;
int State::screenWIDTH = 3072 * 0.6f;
int State::screenHEIGHT = 1920 * 0.6f;

State::State()
{
    // The constructor can be empty if initialization is done in the main function.
}

// DEFINE the destructor here. 
// Now, the compiler has the full definitions of the interfaces and can correctly destroy the unique_ptrs.
State::~State() = default;
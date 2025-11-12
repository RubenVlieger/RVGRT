#include "State.hpp"
#include "platform/Platform.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/Renderer.hpp"

// The global singleton instance
State State::state = State();

State::State()
{
    // The constructor can be empty if initialization is done in the main function.
}

// DEFINE the destructor here. 
// Now, the compiler has the full definitions of the interfaces and can correctly destroy the unique_ptrs.
State::~State() = default;
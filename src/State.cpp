#include "State.hpp"
#include "platform/NetworkClient.hpp"
#include "platform/Platform.hpp"
#include "renderer/GraphicsDevice.hpp"
#include "renderer/Renderer.hpp"

// The global singleton instance
State State::state = State();

int State::dispWIDTH = 1536; // internal
int State::dispHEIGHT = 960;
int State::screenWIDTH = 3072; // external
int State::screenHEIGHT = 1920;

State::State() {
  // The constructor can be empty if initialization is done in the main
  // function.
  Character npc;
  otherCharacters.push_back(npc);

  // Default home is world spawn
  homePosition = glm::vec3(508.0f, 156.0f, 408.0f);
}

// DEFINE the destructor here.
// Now, the compiler has the full definitions of the interfaces and can
// correctly destroy the unique_ptrs.
State::~State() = default;
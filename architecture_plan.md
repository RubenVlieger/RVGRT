Here is the comprehensive 4-phase architecture and implementation plan:

RVGRT: Player Collisions, Terrain Interaction & Block Sync — Implementation Plan

Architecture Overview

The engine currently has:

GPU path tracer with a 3-level SVO (Indirection → Sector → Brick → Voxel) traversed by trace() and traceShadow() in intersections.h
Toroidal streaming (MaterialMap::UpdateStreaming()) that loads/unloads 32³ sectors as the camera moves
Procedural-only terrain — Evaluate() in TerrainGeneration.h and material_gen.shader generate voxels on the fly; there is no mechanism for runtime voxel modification
Server (FastAPI) that broadcasts player state and chat; already has BlockUpdateMessage in models.py but the client only receives block updates via broadcast (doesn't store them)
Player (Character) has no collision detection with voxels — noclip mode is just a stub that prints "not yet implemented"
NetworkClient serializes character transforms but doesn't send/receive block edits
Phase 1: Core Noclip & Player-Terrain Collision

Goal: Toggle between noclip (free flight) and clipped (physics) modes, with collision response against the voxel world.

1.1 — Noclip Command (/noclip)

File changes:

include/State.hpp — add bool noclipMode = true; (default noclip since current gameplay has no collision)
src/console/RegisterCommands.cpp — replace cmd_noclip stub with a toggle:
State::state.noclipMode = !State::state.noclipMode;
Print(console, State::state.noclipMode ? "Noclip enabled" : "Noclip disabled — collision on");
1.2 — CPU-Side Ray Casting Utility

New file: include/VoxelRaycast.hpp

A C++ side reimplementation of the trace() logic from intersections.h (the C++ reference path already exists). This function takes rayPos, rayDir, and accessors to the SVO CPU mirrors (_indirectionCPU, _sectorInfoCPU, occupancy/data CPU mirrors) and returns a hitInfo. This enables:

Collision detection (shoot rays downward from player feet to find ground)
Block placement/removal (shoot ray from camera center to find targeted block)
No GPU round-trip needed — the CPU mirrors already exist in MaterialMap
Key design: Expose MaterialMap::Raycast(pos, dir, maxDist) as a public method. It will:

Convert world-space ray into the toroidal coordinate system using _worldOrigin
Walk the SVO using the CPU-side copies of _indirectionCPU, _sectorInfoCPU, and the brick pool occupancy/data
Return {hit, position, normal, matID, distance} — the voxel coordinate and adjacent air coordinate for block placement logic
File changes:

include/renderer/MaterialMap.hpp — add public method RaycastResult Raycast(simd_float3 pos, simd_float3 dir, float maxDist) const;
src/renderer/MaterialMap.mm — implement using _indirectionCPU and _sectorInfoCPU data
1.3 — Player-Terrain Collision Response

File changes:

include/Character.hpp — add collision-related fields:
bool onGround = false;
float playerHeight = 1.62f; (eye height above feet)
float playerRadius = 0.3f; (horizontal collision radius)
float playerWidth = 0.6f; (bounding box width)
src/Character.cpp — in Character::Update(), when !State::state.noclipMode:
Apply velocity as proposed movement
Cast 4 downward rays from the player's foot corners (±playerRadius in X and Z) to find ground
If ground is within step distance, snap player feet to ground + set onGround = true, zero downward velocity
For horizontal collision: cast rays in the movement direction from the player's bounding box edges; if hit, push the player out of the voxel
When jump is pressed and onGround: apply upward velocity = jumpSpeed
When noclip: no collision checks, free movement (current behavior)
include/State.hpp — when noclipMode changes, reset gravity accordingly
1.4 — Gravity Behavior

Current: gravityAmount defaults to 0.0 with /gravity to set it manually.

New behavior:

When noclip is enabled: gravityAmount is forced to 0
When noclip is disabled: gravityAmount is set to a default (e.g. -9.8 * dt equivalent in the unit system, something like 0.015), and jump key behavior changes from "move up" to actual jump (velocity impulse)
File changes:

src/Character.cpp — In Update(), when noclip disabled, apply gravityAmount as continuous downward acceleration. When jump key pressed and onGround, apply upward impulse.
Phase 2: Terrain Modification (Block Place/Remove) & Ray Casting

Goal: The player can left-click to remove a block and right-click to place one, using camera-centered ray casting into the SVO.

2.1 — Block Interaction State

New file: include/BlockInteraction.hpp

struct BlockEdit {
    int32_t x, y, z;  // World voxel coordinates
    uint8_t matID;     // 0 = air (remove), else material ID (place)
};

enum class BlockAction { Remove, Place };
File changes:

include/State.hpp — add:
std::vector<BlockEdit> localBlockEdits; — edits made by this client (since last reset)
std::mutex blockEditsMutex; — thread-safe access
BlockAction currentBlockAction = BlockAction::Remove;
uint8_t selectedMaterialID = 2; (default: grass)
2.2 — CPU-Side Block Modification on SVO Mirrors

File changes:

include/renderer/MaterialMap.hpp — add public methods:
bool RemoveVoxel(int32_t wx, int32_t wy, int32_t wz); — zero out the voxel in the occupancy mask and data buffer, and propagate the change to GPU
bool PlaceVoxel(int32_t wx, int32_t wy, int32_t wz, uint8_t matID); — set the occupancy bit and material ID, propagate to GPU
void ApplyBlockEdits(const std::vector<BlockEdit>& edits); — batch apply for initial sync
void ResetBlockEdits(); — restore all edited voxels to procedural state
Implementation in src/renderer/MaterialMap.mm:

For each voxel coordinate (wx, wy, wz):
Convert to toroidal sector coordinates using _worldOrigin
Look up SectorState from _sectorStates
Navigate SVO levels: sector → brick → sub-brick → voxel
Remove: Clear the occupancy bit for that voxel; if the sub-brick becomes empty (all bits 0), clear that sub-brick occupancy; propagate upward to brick mask
Place: Set the occupancy bit and write matID to the data buffer
Upload modified occupancy/data sectors to GPU buffers
Track the edit in _appliedEdits list (for reset capability)
This requires adding an _appliedEdits member to MaterialMap that stores the original values so they can be restored on reset.

2.3 — CPU-Side SVO Read Access (for Raycasting)

The raycasting needs read access to occupancy and data buffers. The MaterialMap already has:

_indirectionCPU (vector) — mirror of the indirection texture
_sectorInfoCPU (vector) — mirror of sector buffer
We also need CPU mirrors of the occupancy and data buffers. Currently these only live on the GPU.

Approach: On MaterialMap::GenerateDetailBatch(), after the GPU compute completes, read back the occupancy and data buffers from GPU to CPU mirrors. Alternatively (simpler), evaluate Evaluate() on CPU at the voxel position to determine if a point is solid, without needing the occupancy/data mirrors.

Recommended approach — Hybrid raycast:

Use the CPU indirection texture and sector info (already available) to determine which sector/brick a voxel belongs to
For occupancy checks, use the GPU readback data or evaluate Evaluate() on CPU as a fallback
The TerrainGeneration.h Evaluate() function is pure math — callable from CPU too. For full-detail sectors, use the CPU mirrors; for unloaded sectors, fall back to Evaluate()
2.4 — Input Handling for Block Interaction

File changes:

include/platform/Platform.hpp — add mouse click events:
std::atomic<bool> leftMouseDown; (remove block)
std::atomic<bool> rightMouseDown; (place block)
std::atomic<bool> leftMouseJustPressed; (edge trigger for single clicks)
src/platform/MacOSPlatform.mm — hook mouseDown: / mouseUp: events, set the atomics
src/platform/WindowsPlatform.cpp — hook WM_LBUTTONDOWN/WM_RBUTTONDOWN
2.5 — Block Interaction Logic (Main Loop)

File changes:

src/platform/macos_main.mm — in gameLoop:, after character update:
if (leftMouseJustPressed) {
    auto hit = materialMap.Raycast(character.position, character.direction, 8.0f);
    if (hit.hit && hit.matID != 0) {
        BlockEdit edit{hit.voxelX, hit.voxelY, hit.voxelZ, 0};
        materialMap.RemoveVoxel(edit.x, edit.y, edit.z);
        localBlockEdits.push(edit);
        networkClient->SendBlockEdit(edit);
    }
}
if (rightMouseJustPressed) {
    auto hit = materialMap.Raycast(character.position, character.direction, 8.0f);
    if (hit.hit) {
        // Place at adjacent voxel (hit position + normal direction)
        BlockEdit edit{hit.voxelX + hit.normalX, hit.voxelY + hit.normalY, hit.voxelZ + hit.normalZ, selectedMaterialID};
        materialMap.PlaceVoxel(edit.x, edit.y, edit.z, edit.matID);
        localBlockEdits.push(edit);
        networkClient->SendBlockEdit(edit);
    }
}
2.6 — Visual Feedback: Block Highlighting (Optional Enhancement)

Extend ShaderTypes.h FrameData to include a highlight position:

simd_int3 highlightVoxel;  // World-space voxel being looked at
simd_float3 highlightNormal; // Face normal
float highlightActive;     // 1.0f if highlighting, 0.0f if not
In the GBuffer shader, render this voxel with a subtle wireframe or outline overlay.

Phase 3: Server-Side Block Change Tracking & Broadcast

Goal: The server maintains an authoritative list of all block changes and sends the full list to newly connecting clients, while broadcasting incremental updates in real-time.

3.1 — Server Data Model: Block Change Store

File changes in /RVGRT-server/src/models.py:

Add:

class BlockChangeMessage(BaseMessage):
    type: Literal["block_change"] = "block_change"
    x: int
    y: int
    z: int
    mat_id: int  # 0 = removed, else = material type placed

class BlockSyncMessage(BaseMessage):
    type: Literal["block_sync"] = "block_sync"
    changes: list[dict]  # Full list of {x, y, z, mat_id} dicts

class BlockResetMessage(BaseMessage):
    type: Literal["block_reset"] = "block_reset"

# Update ClientMessage discriminated union to include BlockChangeMessage
3.2 — Server Block Store

File changes in /RVGRT-server/src/server.py:

Add to GameServer:

class GameServer:
    def __init__(self):
        ...
        self.block_changes: list[dict] = []  # Ordered list of all changes
        self.block_change_set: dict[tuple[int,int,int], int] = {}  # {(x,y,z): mat_id} for dedup
New methods:

def add_block_change(self, x, y, z, mat_id): — Append the change to block_changes, update block_change_set[(x,y,z)] = mat_id (later edits to the same voxel overwrite)
def get_block_changes(self) -> list[dict]: — Return the current deduplicated list from block_change_set
def reset_block_changes(self): — Clear both block_changes and block_change_set
Modify broadcast() to also forward block_change messages (rename from the current BlockUpdateMessage type "block" to "block_change" for consistency, or keep "block" and add server tracking — either works).

3.3 — Initial Sync on Connect

File changes in /RVGRT-server/src/main.py:

In websocket_endpoint, after sending the init message, immediately send a block_sync message containing all accumulated block changes:

# After sending init message
sync_msg = BlockSyncMessage(changes=server.get_block_changes())
await websocket.send_text(sync_msg.model_dump_json())
3.4 — Block Change Broadcast

When a client sends a block message (already handled), the server now also:

Calls server.add_block_change(msg.x, msg.y, msg.z, msg.mat_id)
Broadcasts the change to all other clients (already done)
Also sends the change back to the originating client for confirmation (optional — currently excluded)
3.5 — Reset Endpoint

File changes in /RVGRT-server/src/main.py:

Add internal API endpoint:

@app.post("/internal/reset_blocks")
async def reset_blocks():
    server.reset_block_changes()
    # Notify all connected clients
    await server.broadcast(BlockResetMessage().model_dump_json())
    return {"status": "ok", "changes_cleared": True}
3.6 — Logger Admin Reset Button

File changes: /RVGRT-server/src/logger_admin/admin_app/views.py

Add a reset button to the admin view:

Add a form with a POST action to /internal/reset_blocks
The form sits next to the broadcast message form
File changes: /RVGRT-server/src/logger_admin/admin_app/templates/admin_app/logs.html

Add a second form:

<form method="POST" action="{% url 'reset_blocks' %}">
    <button type="submit">Reset All Block Changes</button>
</form>
Add a corresponding URL and view.

File changes: /RVGRT-server/src/logger_admin/admin_app/urls.py

path('reset_blocks/', views.reset_blocks_view, name='reset_blocks'),
And in views.py:

@staff_member_required
def reset_blocks_view(request):
    if request.method == "POST":
        backend_url = os.environ.get("FASTAPI_URL", "http://rvgrt-backend:8000")
        requests.post(f"{backend_url}/internal/reset_blocks", timeout=2)
    return redirect('admin_logs')
Phase 4: Client-Side Block Sync & Reset Command

Goal: The client receives block edits from the server (both initial sync and live updates) and applies them locally. The /reset command triggers a full block change reset.

4.1 — NetworkClient Block Edit Support

File changes: include/platform/NetworkClient.hpp

Add virtual methods:

// Send a block edit to the server
virtual void SendBlockEdit(int32_t x, int32_t y, int32_t z, uint8_t matID) = 0;

// Set callback for receiving block edits from the server
using BlockEditCallback = std::function<void(int32_t x, int32_t y, int32_t z, uint8_t matID)>;
virtual void SetBlockEditCallback(BlockEditCallback callback) = 0;

// Set callback for block sync (initial full list on connect)
using BlockSyncCallback = std::function<void(const std::vector<BlockEdit>& edits)>;
virtual void SetBlockSyncCallback(BlockSyncCallback callback) = 0;

// Set callback for block reset notification
using BlockResetCallback = std::function<void()>;
virtual void SetBlockResetCallback(BlockResetCallback callback) = 0;
File changes: src/platform/MacOSNetworkClient.mm

In the ReadLoop JSON handler, add handlers for:

"block_change" → invoke _blockEditCallback(x, y, z, matID)
"block_sync" → parse the changes array and invoke _blockSyncCallback(edits)
"block_reset" → invoke _blockResetCallback()
Add SendBlockEdit() method that sends:

{"type": "block", "x": 123, "y": 45, "z": 678, "mat_id": 0}
File changes: src/platform/Win32NetworkClient.cpp — same changes (analogous)

4.2 — Wiring Block Edits into Game Loop

File changes: src/platform/macos_main.mm

In applicationDidFinishLaunching:, after setting up the network client, add:

networkClient->SetBlockEditCallback([](int32_t x, int32_t y, int32_t z, uint8_t matID) {
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    BlockEdit edit{x, y, z, matID};
    State::state.pendingRemoteEdits.push(edit);
});

networkClient->SetBlockSyncCallback([](const std::vector<BlockEdit>& edits) {
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    State::state.pendingRemoteEdits = edits; // Replace with full sync
});

networkClient->SetBlockResetCallback([]() {
    // Signal that all block changes should be reset
    State::state.blockResetRequested = true;
});
In gameLoop:, before rendering, drain the edit queues:

{
    std::lock_guard<std::mutex> lock(State::state.blockEditsMutex);
    for (auto& edit : State::state.pendingRemoteEdits) {
        if (edit.matID == 0)
            materialMap->RemoveVoxel(edit.x, edit.y, edit.z);
        else
            materialMap->PlaceVoxel(edit.x, edit.y, edit.z, edit.matID);
    }
    State::state.pendingRemoteEdits.clear();
    
    if (State::state.blockResetRequested) {
        materialMap->ResetBlockEdits();
        State::state.localBlockEdits.clear();
        State::state.blockResetRequested = false;
    }
}
4.3 — /reset Command Update

File changes: src/console/RegisterCommands.cpp

Update cmd_reset to also reset block changes:

static void cmd_reset(const std::vector<std::string>&, GameConsole& console) {
    auto& c = State::state.character;
    c.speed = 0.05f;
    c.sensitivity = 0.00003f;
    c.gravityAmount = 0.0f;
    c.FOV = 70.0f;
    c.jumpSpeed = 2.0f;
    State::state.flyMode = false;
    State::state.noclipMode = true;
    
    // Reset all block changes
    State::state.blockResetRequested = true;
    auto cb = console.GetChatSendCallback();
    if (cb) cb(console.GetPlayerName(), "/reset");
    Print(console, "All settings and block changes reset to defaults.");
}
Also send a message to the server to reset (if connected):

Add callback SendResetCallback to GameConsole or use the NetworkClient directly to send a chat message "/reset" which the server can intercept and clear its block store.
4.4 — In-Game /reset Server-Side Handling

File changes: /RVGRT-server/src/main.py

Intercept chat messages that are exactly "/reset":

elif raw.get("type") == "chat":
    msg = ChatMessage.model_validate(raw)
    if msg.text.strip() == "/reset":
        server.reset_block_changes()
        reset_msg = BlockResetMessage()
        await server.broadcast(reset_msg.model_dump_json())
    else:
        # Regular chat handling
        msg.client_id = client_id
        await server.broadcast(msg.model_dump_json(), exclude=client_id)
4.5 — Block Edit Deduplication on Client

When receiving a block_sync on connect, the client should apply all edits regardless (they may overwrite procedural terrain). The MaterialMap::ApplyBlockEdits() method handles this.

For live updates, apply immediately from the callback queue as described above.

4.6 — Performance Consideration: Block Change List Size

As the number of block changes grows, the block_sync message on connect could become very large. Mitigations:

Deduplication: The server stores {(x,y,z): mat_id}, so the same block edited multiple times only counts once
Compression: If the list grows beyond ~10,000 entries, consider gzip compression on the WebSocket message
The reset button on log.rvgrt.rubenvlieger.nl provides an escape hatch for performance — resetting all changes
Lazy loading: As an optional future enhancement, send changes in chunks (paginated) over multiple WebSocket messages rather than one massive message
Summary of All Files Changed

Phase	File	Changes
1	include/State.hpp	Add noclipMode, blockResetRequested, pendingRemoteEdits, localBlockEdits, blockEditsMutex, selectedMaterialID
1	include/Character.hpp	Add onGround, playerHeight, playerRadius, collision fields
1	src/Character.cpp	Implement collision resolution using raycasting, toggle noclip/gravity
1	src/console/RegisterCommands.cpp	Implement /noclip toggle, update /reset
2	include/BlockInteraction.hpp	New: BlockEdit struct, BlockAction enum
2	include/renderer/MaterialMap.hpp	Add Raycast(), RemoveVoxel(), PlaceVoxel(), ApplyBlockEdits(), ResetBlockEdits(), _appliedEdits, CPU occupancy/data mirrors
2	src/renderer/MaterialMap.mm	Implement voxel modification + CPU readback + raycast
2	include/platform/Platform.hpp	Add mouse click atomics
2	src/platform/MacOSPlatform.mm	Hook mouse clicks
2	include/renderer/ShaderTypes.h	Optionally add highlightVoxel to FrameData
2	src/renderer/kernels/shaders/direct_light.shader	Optional: render block highlight wireframe
3	RVGRT-server/src/models.py	Add BlockChangeMessage, BlockSyncMessage, BlockResetMessage
3	RVGRT-server/src/server.py	Add block_changes list, block_change_set dict, add_block_change(), get_block_changes(), reset_block_changes()
3	RVGRT-server/src/main.py	Send block_sync on connect, track block changes, add /internal/reset_blocks endpoint, intercept /reset chat
3	RVGRT-server/src/logger_admin/admin_app/views.py	Add reset_blocks_view
3	RVGRT-server/src/logger_admin/admin_app/urls.py	Add reset_blocks URL
3	RVGRT-server/src/logger_admin/admin_app/templates/admin_app/logs.html	Add reset button
4	include/platform/NetworkClient.hpp	Add SendBlockEdit(), SetBlockEditCallback(), SetBlockSyncCallback(), SetBlockResetCallback()
4	src/platform/MacOSNetworkClient.mm	Handle block_change, block_sync, block_reset messages; implement SendBlockEdit()
4	src/platform/Win32NetworkClient.cpp	Same as above for Windows
4	src/platform/macos_main.mm	Wire callbacks, drain edit queues in game loop, add mouse click → raycast → block edit logic
Architectural Risks & Mitigations

CPU Raycast Performance: The CPU trace function walks SVO data. For collision (short rays, ~5-10 voxels), this is extremely fast. For block targeting (8-block reach), also fast. No performance concern.

GPU Sync on Block Edit: When modifying a voxel, we must update GPU buffers. Since edits are individual voxels (tiny writes), we can use didModifyRange on the Metal buffer or a small compute dispatch to patch the occupancy mask and data buffer. The overhead per edit is negligible.
Toroidal Wrapping Complexity: Block edits must be stored in world-space coordinates (not wrapped), because _worldOrigin changes as the player moves. When applying an edit, we resolve the current toroidal position of that world-space coordinate. Edits that fall outside the currently loaded region are stored but deferred until the sector streams in.
Race Conditions: Network callbacks arrive on a background thread. The blockEditsMutex in State ensures safe handoff. The game loop drains the queue under the lock before each frame.
Reset Correctness: On /reset, both client and server clear all changes. The client's MaterialMap::ResetBlockEdits() must re-evaluate Evaluate() for every previously-edited voxel to restore procedural terrain. If this is too expensive, store the original values in _appliedEdits and simply writethem back.
The plan is complete above. Do you have any questions about specific phases, want to adjust priorities, or want me to begin implementation of any phase?
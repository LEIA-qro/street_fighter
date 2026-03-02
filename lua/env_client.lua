console.clear()
console.log("Starting Lock-Step Telemetry Script...")

-- ==========================================
-- HARDWARE OPTIMIZATIONS FOR RL TRAINING
-- ==========================================
client.setwindowsize(1)         -- Reduce emulator window to save GPU resources. We won't be rendering anything, so this is purely for performance.
-- client.invisibleemulation(true) -- Takes away any visual rendering overhead, which can significantly boost performance when running headless.
emu.displayvsync(false)         -- Disables V-Sync to allow the emulator to run as fast as possible without being capped by the monitor's refresh rate.
emu.limitframerate(false)       -- Remove any built-in frame rate limits to let the emulator run at maximum speed, which is crucial for faster RL training iterations.
client.displaymessages(false)   -- Disables on-screen text rendering to save CPU cycles
client.SetSoundOn(false)        -- Disables audio processing, which can be a significant CPU drain.

-- ==========================================

-- Hardcode the states directory so Lua knows where to look
local STATES_DIR = "C:\\Users\\Diego Perea\\Documents\\Code\\street_fighter\\states\\"

-- 1. Check if the server was initialized properly via the command line
local port = comm.socketServerGetPort()
if port == nil then
    console.log("ERROR: Socket server not started. Run via Python script.")
    return
end

console.log("Listening on port: " .. port)
comm.socketServerSetTimeout(10) 

-- Implemented frame skipping
local FRAME_SKIP = 4
local step_count = 0 -- Renamed from frame_count to reflect agent steps

-- Initialize Previous Projectile Variables outside the loop
local prev_p1_proj_x = 0
local prev_p2_proj_x = 0

while true do
    -- 1. Read RAM
    local p1_hp = mainmemory.read_u16_be(0x8042)
    local p2_hp = mainmemory.read_u16_be(0x82C2)
    local p1_x  = mainmemory.read_u16_be(0x8006)
    local p2_x  = mainmemory.read_u16_be(0x8358)
    local p1_y  = mainmemory.read_u16_be(0x800A)
    local p2_y  = mainmemory.read_u16_be(0x828A)

    -- We are back on BizHawk 2.8, so the 'bit' library is restored!
    local p1_state_raw = mainmemory.read_u16_be(0x804E)
    local p2_state_raw = mainmemory.read_u16_be(0x82CE)
    local p1_action_id = bit.rshift(p1_state_raw, 8)
    local p2_action_id = bit.rshift(p2_state_raw, 8)

    -- 2. Read RAM: Projectile State & Delta Calculation
    local raw_p1_proj_x = mainmemory.read_u16_be(0x8506)
    local raw_p2_proj_x = mainmemory.read_u16_be(0x8586)
    
    local active_p1_proj_x = -1
    local active_p2_proj_x = -1

    -- If moving, it is active. If frozen, it is dead (-1).
    if raw_p1_proj_x ~= prev_p1_proj_x then
        active_p1_proj_x = raw_p1_proj_x
    end
    
    if raw_p2_proj_x ~= prev_p2_proj_x then
        active_p2_proj_x = raw_p2_proj_x
    end

    -- Update previous states for the next frame's comparison
    prev_p1_proj_x = raw_p1_proj_x
    prev_p2_proj_x = raw_p2_proj_x

    -- 3. Format Payload (Now 10 dimensions) & Send
    local payload = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
        p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_action_id, p2_action_id, active_p1_proj_x, active_p2_proj_x)
    
    comm.socketServerSend(payload)
    
    -- 3. Strict Spinlock: Wait for Python's response before advancing
    local response = ""
    while response == "" or response == nil do
        response = comm.socketServerResponse()
    end

    -- Remove the newline character for clean processing
    response = string.gsub(response, "\n", "")

    -- Check for special RESET command from Python
    if response == "EXIT" then
        console.log("Received EXIT command. Restoring defaults and shutting down...")
        client.setwindowsize(2)        
        -- client.invisibleemulation(false) 
        emu.displayvsync(false)        
        emu.limitframerate(true)       
        client.displaymessages(true)   
        client.SetSoundOn(true)        
        client.exit()                  -- Safely terminates the BizHawk application
        break
        
    elseif string.sub(response, 1, 5) == "RESET" then
        local state_file_path = string.sub(response, 7) -- Extract the state name after "RESET "
        console.log("Received RESET command. Loading New Random State... ")
        savestate.load(state_file_path)
        
        -- Skip input injection and frame advance, yield control to the newly loaded state
        emu.frameadvance() 
    else
        -- Normal Step: Inject Inputs
        local input = {}

        if string.sub(response, 1, 1) == "1" then input["P1 Up"] = true end
        if string.sub(response, 2, 2) == "1" then input["P1 Down"] = true end
        if string.sub(response, 3, 3) == "1" then input["P1 Left"] = true end
        if string.sub(response, 4, 4) == "1" then input["P1 Right"] = true end
        if string.sub(response, 5, 5) == "1" then input["P1 A"] = true end
        if string.sub(response, 6, 6) == "1" then input["P1 B"] = true end
        if string.sub(response, 7, 7) == "1" then input["P1 C"] = true end
        if string.sub(response, 8, 8) == "1" then input["P1 X"] = true end
        if string.sub(response, 9, 9) == "1" then input["P1 Y"] = true end
        if string.sub(response, 10, 10) == "1" then input["P1 Z"] = true end
        -- We sleep the Start and Mode buttons to avoid the agent accidentally pausing or opening the menu
        -- if string.sub(response, 11, 11) == "1" then input["P1 Start"] = true end
        -- if string.sub(response, 12, 12) == "1" then input["P1 Mode"] = true end
        
        -- ACTION REPEAT: Hold the input and advance multiple frames
        for i = 1, FRAME_SKIP do
            joypad.set(input)
            emu.frameadvance()
        end
    end

    -- Debugging
    if step_count % 240 == 0 then -- Responding every 16 seconds
        if response ~= nil and response ~= "" then
            console.log("Python Responding: " .. response)
        end
    end
    
    -- 4. Advance exactly one frame
    step_count = step_count + 1
end


console.clear()
console.log("Starting Lock-Step Telemetry Script...")

-- ==========================================
-- RESTORING HARDWARE OPTIMIZATIONS 
-- ==========================================
emu.limitframerate(true)
client.setwindowsize(4)         
client.invisibleemulation(false) 
emu.displayvsync(false)        
client.displaymessages(true)  
client.SetSoundOn(true)        

-- ==========================================

package.loaded["generated_config"] = nil 
local python_config = require("generated_config")


local STATES_DIR = python_config.STATES_DIR

-- Check if the server was initialized properly via the command line
local port = comm.socketServerGetPort()
if port == nil then
    console.log("ERROR: Socket server not started. Run via Python script.")
    return
end

console.log("Listening on port: " .. port)
comm.socketServerSetTimeout(10) 

-- Implemented frame skipping
local FRAME_SKIP = 4
local step_count = 0 -- Tracks agent steps for debugging

-- Initialize Previous Projectile Variables outside the loop
local prev_p1_proj_x = 0
local prev_p2_proj_x = 0

while true do
    -- Read RAM
    local p1_hp = mainmemory.read_u16_be(0x8042)
    local p2_hp = mainmemory.read_u16_be(0x82C2)
    local p1_x  = mainmemory.read_u16_be(0x8006)
    local p2_x  = mainmemory.read_u16_be(0x8358)
    local p1_y  = mainmemory.read_u16_be(0x800A)
    local p2_y  = mainmemory.read_u16_be(0x828A)

    local p1_state_raw = mainmemory.read_u16_be(0x804E)
    local p2_state_raw = mainmemory.read_u16_be(0x82CE)
    local p1_action_id = bit.rshift(p1_state_raw, 8)
    local p2_action_id = bit.rshift(p2_state_raw, 8)

    -- Read RAM: Projectile State & Delta Calculation
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

    -- Using read_u8 because Character IDs are standard 8-bit integers
    local p1_char_id = mainmemory.read_u8(0x81DA)
    local p2_char_id = mainmemory.read_u8(0x845A)

    -- Format Payload (Now 10 dimensions) & Send
    local payload = string.format("0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
        p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, 
        p1_action_id, p2_action_id, 
        active_p1_proj_x, active_p2_proj_x,
        p1_char_id, p2_char_id)
    
    comm.socketServerSend(payload)
    
    -- Strict Spinlock: Wait for Python's response before advancing
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
        -- Normal Step: Inject Inputs via 20-Bit Protocol
        local p1_input = {}
        local p2_input = {}
        local p1_controlled = false
        local p2_controlled = false

        -- If the string has at least 10 characters, parse Player 1
        if string.len(response) >= 10 then
            local p1_cmd = string.sub(response, 1, 10)
            if p1_cmd ~= ".........." then
                p1_controlled = true
                if string.sub(p1_cmd, 1, 1) == "1" then p1_input["Up"] = true end
                if string.sub(p1_cmd, 2, 2) == "1" then p1_input["Down"] = true end
                if string.sub(p1_cmd, 3, 3) == "1" then p1_input["Left"] = true end
                if string.sub(p1_cmd, 4, 4) == "1" then p1_input["Right"] = true end
                if string.sub(p1_cmd, 5, 5) == "1" then p1_input["A"] = true end
                if string.sub(p1_cmd, 6, 6) == "1" then p1_input["B"] = true end
                if string.sub(p1_cmd, 7, 7) == "1" then p1_input["C"] = true end
                if string.sub(p1_cmd, 8, 8) == "1" then p1_input["X"] = true end
                if string.sub(p1_cmd, 9, 9) == "1" then p1_input["Y"] = true end
                if string.sub(p1_cmd, 10, 10) == "1" then p1_input["Z"] = true end
            end
        end

        -- If the string has 20 characters, parse Player 2
        if string.len(response) >= 20 then
            local p2_cmd = string.sub(response, 11, 20)
            if p2_cmd ~= ".........." then
                p2_controlled = true
                if string.sub(p2_cmd, 1, 1) == "1" then p2_input["Up"] = true end
                if string.sub(p2_cmd, 2, 2) == "1" then p2_input["Down"] = true end
                if string.sub(p2_cmd, 3, 3) == "1" then p2_input["Left"] = true end
                if string.sub(p2_cmd, 4, 4) == "1" then p2_input["Right"] = true end
                if string.sub(p2_cmd, 5, 5) == "1" then p2_input["A"] = true end
                if string.sub(p2_cmd, 6, 6) == "1" then p2_input["B"] = true end
                if string.sub(p2_cmd, 7, 7) == "1" then p2_input["C"] = true end
                if string.sub(p2_cmd, 8, 8) == "1" then p2_input["X"] = true end
                if string.sub(p2_cmd, 9, 9) == "1" then p2_input["Y"] = true end
                if string.sub(p2_cmd, 10, 10) == "1" then p2_input["Z"] = true end
            end
        end
        
        -- ACTION REPEAT: Hold the input and advance multiple frames
        for i = 1, FRAME_SKIP do
            if p1_controlled then joypad.set(p1_input, 1) end
            if p2_controlled then joypad.set(p2_input, 2) end
            emu.frameadvance()
        end
    end

    -- Debugging
    if step_count % 240 == 0 then -- Responding every 16 seconds
        if response ~= nil and response ~= "" then
            console.log("Python Responding: " .. response)
        end
    end
    
    -- Advance exactly one frame
    step_count = step_count + 1
end


console.clear()
console.log("Starting Lock-Step Telemetry Script...")

-- ==========================================
-- RESTORING HARDWARE OPTIMIZATIONS 
-- ==========================================
emu.limitframerate(true)
client.speedmode(100)           -- Force standard 100% speed regardless of training speed
client.setwindowsize(4)         
client.invisibleemulation(false) 
emu.displayvsync(false)        
client.displaymessages(true)  
client.SetSoundOn(true)        

-- ==========================================

package.loaded["generated_config"] = nil 
local python_config = require("generated_config")

-- Input Display configuration (extracted from Input_Display.lua)
local enable_input_display = python_config.ENABLE_INPUT_DISPLAY or false
local p1_label = python_config.P1_MODEL_NAME or "P1"
local p2_label = python_config.P2_MODEL_NAME or "P2"

console.log("Input Display Enabled Flag: " .. tostring(enable_input_display))

local id_xpos, id_ypos = 8, 8
local id_cyp = 0xC0FDDBCF -- Button Yup Pressed
local id_cnp = 0xC0000000 -- Button Not Pressed
local id_cbg = 0xC0000E60 -- Background

local function draw_model_labels()
    -- Draw Player 1 Label (Left)
    gui.text(10, 660, p1_label, 0xFFFFFFFF, "bottomleft")
    -- Draw Player 2 Label (Right)
    gui.text(600, 660, p2_label, 0xFFFFFFFF, "bottomleft")
end

local function draw_inputs()
    if not enable_input_display then return end
    
    for h = 1, 2 do
        local c = joypad.get(h)

        if h == 2 then
            id_xpos = 100
        else
            id_xpos = 8
        end

        -- Check if controller is valid (has at least one standard key)
        if c['Up'] ~= nil or c['A'] ~= nil or c['Start'] ~= nil then
            local x = id_xpos + (h - 1) * 50
            local y = id_ypos
            
            -- Draw controller background
            gui.drawLine(x+ 8,y+ 0,x+37,y+ 0,id_cbg)
            gui.drawLine(x+ 4,y+ 1,x+41,y+ 1,id_cbg)
            gui.drawRectangle(x+ 4,y+ 2,37,11,id_cbg,id_cbg)
            gui.drawRectangle(x+ 0,y+ 4,3,8,id_cbg,id_cbg)
            gui.drawRectangle(x+42,y+ 4,3,8,id_cbg,id_cbg)
            gui.drawRectangle(x+ 1,y+14,10,1,id_cbg,id_cbg)
            gui.drawRectangle(x+ 3,y+16,7,1,id_cbg,id_cbg)
            gui.drawRectangle(x+ 5,y+18,4,1,id_cbg,id_cbg)
            gui.drawRectangle(x+34,y+14,10,1,id_cbg,id_cbg)
            gui.drawRectangle(x+35,y+16,7,1,id_cbg,id_cbg)
            gui.drawRectangle(x+36,y+18,4,1,id_cbg,id_cbg)
            gui.drawRectangle(x+ 2,y+ 2,1,1,id_cbg,id_cbg)
            gui.drawRectangle(x+42,y+ 2,1,1,id_cbg,id_cbg)
            gui.drawLine(x+ 1,y+13,x+ 3,y+13,id_cbg)
            gui.drawLine(x+42,y+13,x+44,y+13,id_cbg)
            
            -- Draw buttons
            gui.drawRectangle(x+ 8,y+ 2, 3, 3,c['Up']    and id_cyp or id_cnp,c['Up']    and id_cyp or id_cnp)
            gui.drawRectangle(x+ 8,y+10, 3, 3,c['Down']  and id_cyp or id_cnp,c['Down']  and id_cyp or id_cnp)
            gui.drawRectangle(x+ 4,y+ 6, 3, 3,c['Left']  and id_cyp or id_cnp,c['Left']  and id_cyp or id_cnp)
            gui.drawRectangle(x+12,y+ 6, 3, 3,c['Right'] and id_cyp or id_cnp,c['Right'] and id_cyp or id_cnp)
            gui.drawRectangle(x+25,y+ 6, 1, 1,c['X']     and id_cyp or id_cnp,c['X']     and id_cyp or id_cnp)
            gui.drawRectangle(x+31,y+ 4, 1, 1,c['Y']     and id_cyp or id_cnp,c['Y']     and id_cyp or id_cnp)
            gui.drawRectangle(x+37,y+ 2, 1, 1,c['Z']     and id_cyp or id_cnp,c['Z']     and id_cyp or id_cnp)
            gui.drawRectangle(x+19,y+ 5, 3, 1,c['Start'] and id_cyp or id_cnp,c['Start'] and id_cyp or id_cnp)
            
            gui.drawEllipse(x+26,y+ 9, 3, 3,c['A'] and id_cyp or id_cnp,c['A'] and id_cyp or id_cnp)
            gui.drawEllipse(x+32,y+ 7, 3, 3,c['B'] and id_cyp or id_cnp,c['B'] and id_cyp or id_cnp)
            gui.drawEllipse(x+38,y+ 5, 3, 3,c['C'] and id_cyp or id_cnp,c['C'] and id_cyp or id_cnp)
        end
    end
end

local STATES_DIR = python_config.STATES_DIR
local state_file_path = python_config.PROJECT_ROOT .. "\\.agent_state"

-- Helper to check if the agent should be running (PLAY) or idle (PAUSE)
local is_running = false -- Default to false for match tests
local frame_counter = 0
local function check_state()
    frame_counter = frame_counter + 1
    -- Throttle disk reads to once every 30 frames to eliminate high-frequency I/O micro-stutters
    if frame_counter % 30 ~= 0 and is_running ~= nil then
        return is_running
    end
    
    local f = io.open(state_file_path, "r")
    if f ~= nil then
        local state = f:read("*all")
        f:close()
        if string.find(state, "PLAY") then return true end
        if string.find(state, "PAUSE") then return false end
    end
    return is_running -- Fallback to current state
end

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
    is_running = check_state()

    if not is_running then
        gui.clearGraphics()
        draw_inputs()
        draw_model_labels()
        emu.frameadvance()
    else
        -- Read RAM
        local p1_hp = mainmemory.read_u16_be(0x8042)
        local p2_hp = mainmemory.read_u16_be(0x82C2)
        local p1_x  = mainmemory.read_u16_be(0x8006)
        local p2_x  = mainmemory.read_u16_be(0x8286)
        local p1_y  = mainmemory.read_u16_be(0x800A)
        local p2_y  = bit.band(mainmemory.read_u16_be(0x828A), 0xFF)

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

        -- Low-byte Character IDs (0..11) and 16-bit Relative Distance
        local p1_char_id = mainmemory.read_u8(0x81DB)
        local p2_char_id = mainmemory.read_u8(0x845B)
        local rel_dist = mainmemory.read_u16_be(0x834C)

        -- Format Payload (Now 13 dimensions) & Send
        local payload = string.format("0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
            p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, 
            p1_action_id, p2_action_id, 
            active_p1_proj_x, active_p2_proj_x,
            p1_char_id, p2_char_id, rel_dist)
        
        comm.socketServerSend(payload)
        
        -- Strict Spinlock: Wait for Python's response before advancing
        local response = ""
        local wait_start_time = os.time() -- Record the exact time we started waiting
        local TIMEOUT_LIMIT = 120 -- Generous 120s limit for interactive menu navigation / profiling
        
        while response == "" or response == nil do
            response = comm.socketServerResponse()
            
            -- THE DEAD MAN'S SWITCH
            -- If current time minus start time exceeds our limit, trigger the failsafe
            if os.difftime(os.time(), wait_start_time) > TIMEOUT_LIMIT then
                console.log("CRITICAL ERROR: No response from Python for " .. TIMEOUT_LIMIT .. " seconds.")
                console.log("Triggering Dead Man's Switch. Shutting down emulator...")
                
                -- Restore safe defaults before crashing
                client.setwindowsize(2)        
                emu.displayvsync(false)        
                emu.limitframerate(true)       
                client.displaymessages(true)   
                client.SetSoundOn(true)        
                
                client.exit() -- Force close BizHawk
                return        -- Kill the Lua script
            end
        end

        -- Remove the newline character for clean processing
        response = string.gsub(response, "\n", "")

        -- Extract the actual payload by stripping the '{len} ' prefix sent by Python
        local payload_str = string.match(response, "^%d+%s+(.*)$")
        if payload_str then
            response = payload_str
        end

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
            gui.clearGraphics()
            draw_inputs()
            draw_model_labels()
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
                gui.clearGraphics()
                draw_inputs()
                draw_model_labels()
                emu.frameadvance()
            end
        end

        -- Debugging
        if step_count % 240 == 0 then -- Responding every 16 seconds
            if response ~= nil and response ~= "" then
                console.log("Python Responding: " .. response)
            end
        end
        
        -- Periodic memory and console buffer cleanup
        if step_count % 5000 == 0 then
            console.clear()
            collectgarbage("collect")
        end

        -- Advance exactly one frame
        step_count = step_count + 1
    end
end


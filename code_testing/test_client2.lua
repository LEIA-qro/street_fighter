console.clear()
console.log("Starting Lock-Step Telemetry Script...")

-- 1. Check if the server was initialized properly via the command line
local port = comm.socketServerGetPort()
if port == nil then
    console.log("ERROR: Socket server not started. Run via Python script.")
    return
end

console.log("Listening on port: " .. port)
comm.socketServerSetTimeout(10) 

-- 2. Main Loop: Read RAM, Send to Python, Wait for Response, Advance Frame
local frame_count = 0

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

    -- 2. Format Payload & Send
    local payload = string.format("%d,%d,%d,%d,%d,%d,%d,%d\n", 
        p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_action_id, p2_action_id)
    
    comm.socketServerSend(payload)
    
    -- 3. Strict Spinlock: Wait for Python's response before advancing
    local response = comm.socketServerResponse()

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
    -- if string.sub(response, 11, 11) == "1" then input["P1 Start"] = true end
    -- if string.sub(response, 12, 12) == "1" then input["P1 Mode"] = true end
    
    -- Inject the inputs into the emulator core
    joypad.set(input)



    -- Debugging
    if frame_count % 60 == 0 then
        if response ~= nil and response ~= "" then
            console.log("Python says: " .. response)
        end
    end

    
    -- 4. Advance exactly one frame
    frame_count = frame_count + 1
    emu.frameadvance()
end
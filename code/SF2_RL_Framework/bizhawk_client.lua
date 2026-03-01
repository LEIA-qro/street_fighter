---@diagnostic disable: undefined-global

-- 1. Import .NET System Libraries via NLua
local luanet = require("luanet")
luanet.load_assembly("System")

local TcpClient = luanet.import_type("System.Net.Sockets.TcpClient")
local StreamReader = luanet.import_type("System.IO.StreamReader")
local StreamWriter = luanet.import_type("System.IO.StreamWriter")

print("Attempting to connect to Python Server on port 1337...")

-- 2. Safely attempt to open the TCP Socket
local client
local success, err = pcall(function()
    client = TcpClient("127.0.0.1", 1337)
end)

if not success or client == nil then
    print("CRITICAL: Failed to connect to Python.")
    print("Ensure train_ppo.py is running BEFORE starting this script!")
    print("Error details: " .. tostring(err))
    return
end

print("Connected to Python AI!")

-- 3. Wrap the socket in C# Streams for easy Line-by-Line reading
local stream = client:GetStream()
local reader = StreamReader(stream)
local writer = StreamWriter(stream)
writer.AutoFlush = true -- Forces data to send immediately over TCP without waiting

client.speedmode(6400)
client.displaymessages(false)

while true do
    -- Read RAM
    local p1_hp = memory.read_u16_be(0x8042)
    local p2_hp = memory.read_u16_be(0x82C2)
    local p1_x  = memory.read_u16_be(0x8006)
    local p2_x  = memory.read_u16_be(0x8358)
    local p1_y  = memory.read_u16_be(0x800A)
    local p2_y  = memory.read_u16_be(0x828A)
    
    local p1_state_raw = memory.read_u16_be(0x804E)
    local p2_state_raw = memory.read_u16_be(0x82CE)
    local p1_action_id = p1_state_raw >> 8
    local p2_action_id = p2_state_raw >> 8

    -- Format Payload
    local payload = string.format("%d,%d,%d,%d,%d,%d,%d,%d", 
        p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_action_id, p2_action_id)
    
    -- Send using C# StreamWriter. 
    -- WriteLine automatically appends \r\n, which perfectly matches Python's .strip()
    writer:WriteLine(payload)

    -- Wait for Action from Python (Blocking)
    local action_str = reader:ReadLine()
    
    if action_str == "RESET" then
        -- NOTE: Update this absolute path to point to where your state is saved!
        local state_path = "C:\\Users\\Diego Perea\\Documents\\Apps\\BizHawk-2.11-win-x64\\StartRound.State"
        savestate.load(state_path)
    elseif action_str ~= nil then
        -- Parse 12-bit action string and press buttons
        local input = {}
        if string.sub(action_str, 1, 1) == "1" then input["P1 Up"] = true end
        if string.sub(action_str, 2, 2) == "1" then input["P1 Down"] = true end
        if string.sub(action_str, 3, 3) == "1" then input["P1 Left"] = true end
        if string.sub(action_str, 4, 4) == "1" then input["P1 Right"] = true end
        if string.sub(action_str, 5, 5) == "1" then input["P1 A"] = true end
        if string.sub(action_str, 6, 6) == "1" then input["P1 B"] = true end
        if string.sub(action_str, 7, 7) == "1" then input["P1 C"] = true end
        if string.sub(action_str, 8, 8) == "1" then input["P1 X"] = true end
        if string.sub(action_str, 9, 9) == "1" then input["P1 Y"] = true end
        if string.sub(action_str, 10, 10) == "1" then input["P1 Z"] = true end
        if string.sub(action_str, 11, 11) == "1" then input["P1 Start"] = true end
        if string.sub(action_str, 12, 12) == "1" then input["P1 Mode"] = true end
        
        joypad.set(input)
        emu.frameadvance()
    else
        print("Connection lost.")
        break
    end
end
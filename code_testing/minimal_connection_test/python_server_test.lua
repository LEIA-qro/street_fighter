console.clear()
console.log("Starting minimal connection script...")

-- 1. Check if the server was initialized properly via the command line
local port = comm.socketServerGetPort()
if port == nil then
    console.log("ERROR: Socket server not started. Run via Python script.")
    return
end

console.log("Listening on port: " .. port)
comm.socketServerSetTimeout(50)

local frame_count = 0

while true do
    -- 2. Send a PING every 60 frames
    if frame_count % 60 == 0 then
        comm.socketServerSend("PING from BizHawk!\n")
    end
    
    -- 3. Read the PONG from Python
    local response = comm.socketServerResponse()
    if response ~= nil and response ~= "" then
        console.log("Python says: " .. response)
    end
    
    frame_count = frame_count + 1
    emu.frameadvance()
end
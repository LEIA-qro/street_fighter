-- Use mainmemory to avoid domain switching headaches
    

while true do

    -- Read RAM --
    -- HP
    local p1_hp = mainmemory.read_u16_be(0x8042)
    local p2_hp = mainmemory.read_u16_be(0x82C2)
    -- X/Y Coordinates
    local p1_x  = mainmemory.read_u16_be(0x8006)
    local p2_x  = mainmemory.read_u16_be(0x8358)
    local p1_y  = mainmemory.read_u16_be(0x800A)
    local p2_y  = mainmemory.read_u16_be(0x828A)
    -- Action State Comes in HEX
    local p1_state_raw = mainmemory.read_u16_be(0x804E)
    local p2_state_raw = mainmemory.read_u16_be(0x82CE)

    -- Command to make an action. 
    joypad.set({Right = true}, 1)

    -- Crucial: Advance the emulator or BizHawk will hang
    emu.frameadvance()

    if p1_hp ~= mainmemory.read_u16_be(0x8042) then
        print(string.format("P1 HP changed: %d -> %d", p1_hp, mainmemory.read_u16_be(0x8042)))
    end

end
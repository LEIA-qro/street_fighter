-- stand_env_client.lua -- cliente BizHawk del MODO STAND (humano vs IA).
--
-- Clon de match_test_env_client.lua con UNA diferencia de fondo: el payload
-- manda las 25 variables de RAM del data.json de stable-retro, CRUDAS y en
-- orden fijo, para que Python arme la observacion v4 con exactamente el
-- mismo assemble_v4_frame del entrenamiento (paridad bit a bit; el -1 de
-- proyectiles y la mascara de p2_y los deriva Python, no este script).
-- La inyeccion es la de siempre: 20 chars, 10 por jugador; ".........."
-- significa NO inyectar a ese jugador -> su control fisico pasa directo.
-- El stand manda bits del modelo en P1 y puntos en P2: el visitante juega
-- con el control USB configurado como Player 2 en BizHawk.

console.clear()
console.log("Starting LEIA Stand Client (humano vs IA)...")

emu.limitframerate(true)
client.speedmode(100)
client.setwindowsize(4)
client.invisibleemulation(false)
emu.displayvsync(false)
client.displaymessages(true)
client.SetSoundOn(true)

package.loaded["generated_config"] = nil
local python_config = require("generated_config")

local p1_label = python_config.P1_MODEL_NAME or "LEIA IA"
local p2_label = python_config.P2_MODEL_NAME or "RETADOR"

-- Marcador (formato RESET <state>|p1|p2, igual que el match test)
local show_scoreboard = false
local p1_score = 0
local p2_score = 0
local current_match_state = "Manual / Menu"

local function get_state_display_name(path)
    if path == nil or path == "" then return "Manual / Menu" end
    local name = path:match("^.+\\(.+)$") or path:match("^.+/(.+)$") or path
    return string.gsub(name, "%.State$", ""):gsub("%.state$", "")
end

local function draw_overlay()
    if show_scoreboard then
        gui.text(10, 24, p1_label, 0xFFFFFFFF, "bottomleft")
        gui.text(10, 8, string.format("Wins: %d", p1_score), 0xFF00FF00, "bottomleft")
        gui.text(10, 24, p2_label, 0xFFFFFFFF, "bottomright")
        gui.text(10, 8, string.format("Wins: %d", p2_score), 0xFF00FF00, "bottomright")
    else
        gui.text(10, 10, p1_label, 0xFFFFFFFF, "bottomleft")
        gui.text(10, 10, p2_label, 0xFFFFFFFF, "bottomright")
    end
    if current_match_state and current_match_state ~= "None" then
        gui.text(10, 10, current_match_state, 0xFFFFFFFF, "topright")
    end
end

local state_file_path = python_config.PROJECT_ROOT .. "\\.agent_state"
local is_running = false
local frame_counter = 0
local function check_state()
    frame_counter = frame_counter + 1
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
    return is_running
end

local port = comm.socketServerGetPort()
if port == nil then
    console.log("ERROR: Socket server not started. Run via Python script.")
    return
end
console.log("Listening on port: " .. port)
comm.socketServerSetTimeout(10)

local FRAME_SKIP = 4
local step_count = 0

while true do
    is_running = check_state()

    if not is_running then
        gui.clearGraphics()
        draw_overlay()
        emu.frameadvance()
    else
        -- Las 25 variables del data.json, crudas, en el orden que
        -- stand_leia.py declara en PAYLOAD_KEYS (generado de data.json:
        -- offset = address - 0xFF0000; >i2=s16, >u2=u16, |u1=u8).
        local p1_hp = mainmemory.read_s16_be(0x8042)
        local p2_hp = mainmemory.read_s16_be(0x82C2)
        local p1_x = mainmemory.read_u16_be(0x8006)
        local p2_x = mainmemory.read_u16_be(0x8286)
        local p1_y = mainmemory.read_u16_be(0x800A)
        local p2_y = mainmemory.read_u16_be(0x828A)
        local p1_state_word = mainmemory.read_u16_be(0x804E)
        local p2_state_word = mainmemory.read_u16_be(0x82CE)
        local p1_proj_x = mainmemory.read_u16_be(0x8506)
        local p2_proj_x = mainmemory.read_u16_be(0x8586)
        local p1_char = mainmemory.read_u8(0x81DB)
        local p2_char = mainmemory.read_u8(0x845B)
        local rel_dist = mainmemory.read_u16_be(0x834C)
        local p1_btn = mainmemory.read_u8(0x81E2)
        local p2_btn = mainmemory.read_u8(0x845E)
        local p1_air_raw = mainmemory.read_u16_be(0x80C0)
        local p2_air_raw = mainmemory.read_u16_be(0x86F4)
        local rel_y_dist = mainmemory.read_u16_be(0x834E)
        local p1_chest = mainmemory.read_u16_be(0x80DC)
        local p1_head = mainmemory.read_u16_be(0x80E0)
        local p2_chest = mainmemory.read_u16_be(0x835C)
        local p2_head = mainmemory.read_u16_be(0x8360)
        local matches_won = mainmemory.read_u8(0x81DA)
        local enemy_matches_won = mainmemory.read_u8(0x845A)
        local round_timer = mainmemory.read_u8(0x972A)

        local payload = string.format(
            "0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
            p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y,
            p1_state_word, p2_state_word,
            p1_proj_x, p2_proj_x, p1_char, p2_char, rel_dist,
            p1_btn, p2_btn, p1_air_raw, p2_air_raw, rel_y_dist,
            p1_chest, p1_head, p2_chest, p2_head,
            matches_won, enemy_matches_won, round_timer)

        comm.socketServerSend(payload)

        local response = ""
        local wait_start_time = os.time()
        local TIMEOUT_LIMIT = 120

        while response == "" or response == nil do
            response = comm.socketServerResponse()
            if os.difftime(os.time(), wait_start_time) > TIMEOUT_LIMIT then
                console.log("CRITICAL ERROR: No response from Python for " .. TIMEOUT_LIMIT .. " seconds.")
                console.log("Triggering Dead Man's Switch. Shutting down emulator...")
                client.setwindowsize(2)
                emu.displayvsync(false)
                emu.limitframerate(true)
                client.displaymessages(true)
                client.SetSoundOn(true)
                client.exit()
                return
            end
        end

        response = string.gsub(response, "\n", "")
        local payload_str = string.match(response, "^%d+%s+(.*)$")
        if payload_str then
            response = payload_str
        end

        if response == "EXIT" then
            console.log("Received EXIT command. Restoring defaults and shutting down...")
            client.setwindowsize(2)
            emu.displayvsync(false)
            emu.limitframerate(true)
            client.displaymessages(true)
            client.SetSoundOn(true)
            client.exit()
            break
        elseif string.sub(response, 1, 5) == "RESET" then
            local reset_arg = string.sub(response, 7)
            local reset_path = reset_arg
            local sep_idx = string.find(reset_arg, "|")
            if sep_idx then
                reset_path = string.sub(reset_arg, 1, sep_idx - 1)
                local score_str = string.sub(reset_arg, sep_idx + 1)
                local sep2_idx = string.find(score_str, "|")
                if sep2_idx then
                    p1_score = tonumber(string.sub(score_str, 1, sep2_idx - 1)) or 0
                    p2_score = tonumber(string.sub(score_str, sep2_idx + 1)) or 0
                    show_scoreboard = true
                else
                    p1_score = tonumber(score_str) or 0
                    p2_score = 0
                    show_scoreboard = true
                end
            else
                show_scoreboard = false
            end

            console.log("Received RESET command. Loading State: " .. reset_path)
            savestate.load(reset_path)
            current_match_state = get_state_display_name(reset_path)
            gui.clearGraphics()
            draw_overlay()
            emu.frameadvance()
        else
            -- Inyeccion 20 chars: 10 P1 + 10 P2; ".........." = passthrough
            -- (el control fisico de ese puerto manda). El stand siempre manda
            -- puntos en P2: ahi vive el humano.
            local p1_input = {}
            local p2_input = {}
            local p1_controlled = false
            local p2_controlled = false

            if string.len(response) >= 10 then
                local p1_cmd = string.sub(response, 1, 10)
                if p1_cmd ~= ".........." then
                    p1_controlled = true
                    -- Asignacion INCONDICIONAL (true fuerza, false SUELTA):
                    -- joypad.set con un boton ausente lo deja vivo al input
                    -- fisico, y el pad de la IA debe ser suyo COMPLETO --
                    -- un teclado ligado a P1 no puede meterle inputs, ni
                    -- un Start pausar el juego a media demo.
                    p1_input["Up"]    = (string.sub(p1_cmd, 1, 1) == "1")
                    p1_input["Down"]  = (string.sub(p1_cmd, 2, 2) == "1")
                    p1_input["Left"]  = (string.sub(p1_cmd, 3, 3) == "1")
                    p1_input["Right"] = (string.sub(p1_cmd, 4, 4) == "1")
                    p1_input["A"]     = (string.sub(p1_cmd, 5, 5) == "1")
                    p1_input["B"]     = (string.sub(p1_cmd, 6, 6) == "1")
                    p1_input["C"]     = (string.sub(p1_cmd, 7, 7) == "1")
                    p1_input["X"]     = (string.sub(p1_cmd, 8, 8) == "1")
                    p1_input["Y"]     = (string.sub(p1_cmd, 9, 9) == "1")
                    p1_input["Z"]     = (string.sub(p1_cmd, 10, 10) == "1")
                    p1_input["Start"] = false
                    p1_input["Mode"]  = false
                end
            end

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

            for i = 1, FRAME_SKIP do
                if p1_controlled then joypad.set(p1_input, 1) end
                if p2_controlled then joypad.set(p2_input, 2) end
                gui.clearGraphics()
                draw_overlay()
                emu.frameadvance()
            end
        end

        if step_count % 5000 == 0 then
            console.clear()
            collectgarbage("collect")
        end

        step_count = step_count + 1
    end
end

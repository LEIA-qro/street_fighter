
local STATES_MODULE = {}

function STATES_MODULE.get_DIR()
    local handle = io.popen("cd")
    local CURRENT_WORKING_DIR = handle:read("*a"):gsub("\n", ""):gsub("\r", "")
    handle:close()

    -- Grab everything from the start of the string up to and including "street_fighter"
    local PROJECT_ROOT = CURRENT_WORKING_DIR:match("(.*street_fighter)")

    -- Failsafe in case the script is run completely outside the project folder
    if not PROJECT_ROOT then
        console.log("ERROR: Could not find 'street_fighter' in the current path.")
        return nil 
    end

    local final_states_path = PROJECT_ROOT .. "\\states\\"

    local function exists(path)
    local ok, err, code = os.rename(path, path)
    if not ok then
        if code == 13 then -- Permission denied, but it exists
            return true
        end
        return false
    end
    return true
    end

    if not exists(final_states_path) then
        console.log("El directorio de estados no existe.")
    else
        console.log("El directorio de estados ya existe en: " .. final_states_path)
    end
    return final_states_path
end

return STATES_MODULE






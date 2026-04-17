

-- Import the module
local directoryTools = require("testing") 

-- Call the function and catch the returned string
local myStatesDir = directoryTools.get_DIR()



console.clear()


local source = debug.getinfo(1, "S").source

console.log("Source of the current script: " .. source)


console.log(myStatesDir)
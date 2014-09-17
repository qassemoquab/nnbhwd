package = "nnbhwd"
version = "scm-1"

source = {
   url = "git://github.com/qassemoquab/nnbhwd",
   tag = "master"
}

description = {
   summary = "Neural networks additional home-made packages",
   detailed = [[
   	    Stuff we do at office #42
   ]],
   homepage = "https://github.com/qassemoquab/nnbhwd"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}

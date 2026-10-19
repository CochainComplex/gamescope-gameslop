#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include "framegen/net_profile.hpp"

namespace vapor {
namespace fs = std::filesystem;
// Launch settings are separate from learned .bin files. The menu never writes
// weights and deliberately does not mutate a running renderer's GPU topology.
inline const std::map<std::string, std::vector<std::string>> choices{
    {"framegen", {"on", "off"}},
    {"mode", {"motion", "extrapolate", "blend"}},
    {"pipeline", {"warp", "checked", "learned", "predict", "guided", "spatial"}},
    {"multiplier", {"2", "3", "4"}},
    {"online", {"on", "off"}},
    {"training_device", {"gpu", "npu"}},
    {"bidir", {"off", "on"}},
    {"hud", {"1", "2", "0"}},
    {"corner", {"br", "bl", "tr", "tl", "bc"}},
    {"style", {"quiet", "mallsoft", "cga"}},
    {"hud_opacity", {"auto", "0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"}},
    {"hud_scale", {"auto", "1", "2", "3", "4", "5", "6"}},
};
// Menu rows (index into main.cpp's labels) whose LEFT/RIGHT edits one profile
// key. Every other focusable row is an action, the preset selector (1) or the
// unavailable NPU switch (17). Store::editRow is the only edit dispatch.
inline const std::map<int, std::string> menuRowKeys{
    {2, "pipeline"}, {3, "multiplier"}, {4, "mode"}, {5, "framegen"},
    {6, "online"}, {7, "bidir"}, {8, "net_profile"}, {9, "hud"}, {10, "corner"},
    {14, "style"}, {15, "hud_opacity"}, {16, "hud_scale"},
};
inline bool identifier(const std::string &s) {
    return !s.empty() && s.size()<=64 && s!="." && s!=".." &&
        std::all_of(s.begin(),s.end(),[](unsigned char c) {
            return (c>='a' && c<='z') || (c>='A' && c<='Z') ||
                (c>='0' && c<='9') || c=='-' || c=='_' || c=='.';
        });
}
struct Context {
    // Passed by the launcher, before Gamescope replaces DISPLAY/WAYLAND_DISPLAY.
    std::string backend="unknown";
    bool base=false, vrrHybrid=false;
    std::string orientation="normal", outputs="SingleApplication";
    std::vector<std::string> pipelines=choices.at("pipeline");
    bool supportsPipeline(const std::string &name) const {
        return std::find(pipelines.begin(),pipelines.end(),name)!=pipelines.end();
    }
    std::string framegenReason() const {
        if (backend=="headless" || backend=="openvr") return "This backend cannot present generated frames";
        if (orientation!="normal") return "Output rotation makes frame generation dormant";
        if (outputs!="SingleApplication" && outputs!="SteamControlled")
            return "Frame generation requires a single output";
        return "";
    }
    std::string bidirReason() const {
        if (!framegenReason().empty()) return framegenReason();
        if (backend=="sdl") return "SDL cannot present bidirectional frames";
        if (base) return "Base-layer scenes use causal generation";
        if (vrrHybrid) return "VRR hybrid requires causal generation";
        return "";
    }
};
inline void validateLearned(const fs::path &path) {
    if (path.empty() || !fs::exists(path)) return;
    const auto fail=[&] { throw std::runtime_error("Incompatible training file / choose valid GSFP or new path"); };
    if (!fs::is_regular_file(path) || fs::file_size(path)>gamescope::framegen::k_uFramegenNetProfileMaxBytes) fail();
    std::ifstream in(path,std::ios::binary);
    std::vector<uint8_t> bytes(gamescope::framegen::k_uFramegenNetProfileMaxBytes+1);
    in.read(reinterpret_cast<char *>(bytes.data()),bytes.size());
    bytes.resize(static_cast<size_t>(in.gcount()));
    if (in.bad() || !gamescope::framegen::decode_net_profile(bytes)) fail();
}
struct Profile {
    std::map<std::string,std::string> values{
        {"framegen","on"}, {"mode","motion"}, {"pipeline","learned"},
        {"training_device","gpu"}, {"multiplier","2"}, {"online","on"}, {"bidir","off"},
        {"hud","1"}, {"corner","br"}, {"net_profile",""}, {"style","quiet"}, {"hud_opacity","auto"}, {"hud_scale","auto"},
    };
    void set(const std::string &key, const std::string &value) {
        if (key=="net_profile") {
            if (value.size()>4096 || value.find_first_of("\r\n\0",0,3)!=std::string::npos)
                throw std::runtime_error("Invalid learned-profile path");
        } else {
            auto it=choices.find(key);
            if (it==choices.end() || std::find(it->second.begin(),it->second.end(),value)==it->second.end())
                throw std::runtime_error("Invalid profile setting: "+key);
        }
        values[key]=value;
    }
    void cycle(const std::string &key, int delta) {
        const auto &list=choices.at(key);
        int at=int(std::find(list.begin(),list.end(),values.at(key))-list.begin());
        values[key]=list[(at+int(list.size())+delta)%int(list.size())];
    }
    bool motion() const { return values.at("framegen")=="on" && values.at("mode")=="motion"; }
    bool usesNet() const {
        auto &p=values.at("pipeline");
        return motion() && (p=="learned" || p=="predict" || p=="guided" || p=="spatial");
    }
    std::string bidirReason(const Context &context={}) const {
        if (!motion()) return "Bidirectional needs active motion mode";
        if (values.at("pipeline")=="spatial") return "Spatial is causal only";
        return context.bidirReason();
    }
    std::string disabledReason(const std::string &key, const Context &context={}) const {
        if (key=="style") return "";
        if (key=="framegen") return values.at("framegen")=="on" ? "" : context.framegenReason();
        if (values.at("framegen")=="off") return "Enable frame generation first";
        if (key=="pipeline" && !motion()) return "Pipelines apply to motion mode only";
        if ((key=="online" || key=="net_profile") && !usesNet())
            return "Learned profiles need a learned motion pipeline";
        if (key=="bidir") return values.at("bidir")=="on" ? "" : bidirReason(context);
        if ((key=="corner" || key=="hud_opacity" || key=="hud_scale") && values.at("hud")=="0")
            return "Enable the HUD first";
        return "";
    }
    void validate(const Context &context={}) const {
        if (values.at("framegen")=="off") return; // Retain inactive preferences.
        auto reason=context.framegenReason();
        if (!reason.empty()) throw std::runtime_error(reason);
        if (motion() && !context.supportsPipeline(values.at("pipeline")))
            throw std::runtime_error("Selected host does not support this pipeline");
        if (values.at("bidir")=="on" && !(reason=bidirReason(context)).empty())
            throw std::runtime_error(reason);
    }
};
inline Profile load(const fs::path &path) {
    Profile p;
    if (!fs::exists(path)) {
        if (path.stem()=="spatial-experimental") p.set("pipeline","spatial");
        return p;
    }
    if (fs::is_symlink(path) || !fs::is_regular_file(path) || fs::file_size(path)>8192)
        throw std::runtime_error("Profile must be a small regular file");
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read profile");
    std::string line;
    bool version=false;
    std::vector<std::string> seen;
    while (std::getline(in,line)) {
        if (line.empty() || line[0]=='#') continue;
        auto eq=line.find('=');
        if (eq==std::string::npos) throw std::runtime_error("Invalid profile line");
        auto key=line.substr(0,eq), value=line.substr(eq+1);
        if (std::find(seen.begin(),seen.end(),key)!=seen.end()) throw std::runtime_error("Duplicate profile field");
        seen.push_back(key);
        if (key=="version") { if (value!="1") throw std::runtime_error("Unsupported profile version"); version=true; }
        else p.set(key,value);
    }
    if (!version) throw std::runtime_error("Missing profile version");
    p.validate();
    return p;
}
inline void writeAtomic(const fs::path &path, const std::string &bytes) {
    fs::create_directories(path.parent_path());
    std::string temporary=path.string()+".tmp-XXXXXX";
    int fd=mkstemp(temporary.data());
    if (fd<0) throw std::system_error(errno,std::generic_category(),"Create profile staging file");
    try {
        size_t offset=0;
        while (offset<bytes.size()) {
            ssize_t n=write(fd,bytes.data()+offset,bytes.size()-offset);
            if (n<0 && errno==EINTR) continue;
            if (n<=0) throw std::system_error(errno,std::generic_category(),"Write profile");
            offset+=size_t(n);
        }
        if (fsync(fd)!=0) throw std::system_error(errno,std::generic_category(),"Sync profile");
        int result=close(fd); fd=-1;
        if (result!=0) throw std::system_error(errno,std::generic_category(),"Close profile");
        fs::rename(temporary,path);
    } catch (...) { if (fd>=0) close(fd); unlink(temporary.c_str()); throw; }
}
inline void save(const fs::path &path, const Profile &p) {
    if (p.values.at("training_device")!="gpu")
        throw std::runtime_error("NPU training unavailable / offline inference only");
    // Validate even when the caller constructed values directly.
    Profile checked;
    for (auto &[key,value] : p.values) checked.set(key,value);
    checked.validate();
    if (checked.usesNet()) validateLearned(checked.values.at("net_profile"));
    std::string bytes="version=1\n";
    for (auto &[key,value] : checked.values) bytes+=key+"="+value+"\n";
    writeAtomic(path,bytes);
}
inline std::string active(const fs::path &directory) {
    auto path=directory/".active";
    if (!fs::exists(path)) return "default";
    if (fs::is_symlink(path) || !fs::is_regular_file(path) || fs::file_size(path)>65)
        throw std::runtime_error("Invalid active preset");
    std::ifstream in(path); std::string name;
    if (!std::getline(in,name) || !identifier(name) || in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Invalid active preset");
    return name;
}
struct Store {
    fs::path directory;
    std::vector<std::string> names;
    std::vector<std::string> learnedPaths;
    std::string name="default", status="Saved settings apply next launch";
    Profile draft;
    Context context;
    bool dirty=false;
    static std::string environment(const char *key, const std::string &fallback) {
        auto value=std::getenv(key); return value && *value ? value : fallback;
    }
    Store() {
        context.backend=environment("VAPOR_BACKEND","unknown");
        context.base=environment("VAPOR_BASE","0")=="1";
        context.vrrHybrid=environment("VAPOR_VRR_HYBRID","0")=="1";
        context.orientation=environment("VAPOR_ORIENTATION","normal");
        context.outputs=environment("VAPOR_OUTPUTS","SingleApplication");
        auto supported=environment("VAPOR_PIPELINES","");
        if (!supported.empty()) {
            context.pipelines.clear(); std::istringstream input(supported); std::string name;
            while (std::getline(input,name,',')) {
                if (std::find(choices.at("pipeline").begin(),choices.at("pipeline").end(),name)==choices.at("pipeline").end())
                    throw std::runtime_error("Invalid host pipeline capability");
                context.pipelines.push_back(name);
            }
        }
        auto game=environment("VAPOR_GAME","default");
        if (!identifier(game)) throw std::runtime_error("Invalid game identifier");
        auto config=environment("XDG_CONFIG_HOME",environment("HOME",".")+"/.config");
        directory=fs::path(environment("VAPOR_PROFILE_ROOT",config+"/gamescope/profiles"))/game;
        name=environment("VAPOR_PROFILE","");
        if (name.empty()) name=active(directory);
        if (!identifier(name)) throw std::runtime_error("Invalid preset identifier");
        refresh(); draft=load(directory/(name+".ini"));
        // The launcher resolves explicit overrides before starting the host.
        // Show the same file the renderer uses, even with an existing preset.
        auto actual=environment("GAMESCOPE_FRAMEGEN_NET_PROFILE","");
        if (!actual.empty() && actual!=draft.values["net_profile"]) {
            dirty=!draft.values["net_profile"].empty();
            draft.set("net_profile",actual);
            if (dirty) status="Launch file override / save to keep next launch";
        }
        if (draft.values["net_profile"].empty())
            draft.set("net_profile",(directory/"learned.bin").string());
        refreshLearned();
    }
    void edit(const std::string &key, int delta) {
        auto reason=draft.disabledReason(key,context);
        if (!reason.empty()) { status=reason; return; }
        draft.cycle(key,delta); dirty=true;
        if (key=="pipeline") {
            for (size_t i=0; i<choices.at(key).size() && !context.supportsPipeline(draft.values.at(key)); ++i)
                draft.cycle(key,delta);
        }
        status="Unsaved / changes apply next launch";
        if (key=="bidir" && draft.values.at("bidir")=="on")
            status="Bidir adds real-frame delay / next launch";
        if (draft.motion() && !context.supportsPipeline(draft.values.at("pipeline"))) {
            if (context.pipelines.empty()) throw std::runtime_error("Host has no supported motion pipelines");
            draft.set("pipeline",context.pipelines.front());
            status="Pipeline changed / unsupported by selected host";
        }
        // A deliberate parent-setting change also retires incompatible bidir.
        // Training/pipeline preferences remain remembered while inactive.
        if (draft.values.at("bidir")=="on" && !draft.bidirReason(context).empty()) {
            draft.set("bidir","off");
            status="Bidir disabled / "+draft.bidirReason(context);
        }
    }
    // One menu row's LEFT/RIGHT. Rows without a profile key do nothing here.
    void editRow(int row, int delta) {
        auto entry=menuRowKeys.find(row);
        if (entry==menuRowKeys.end()) return;
        const auto &key=entry->second;
        if (key=="net_profile") { cycleLearned(delta); return; }
        if (key=="style") {
            // Menu appearance too: always editable, no dependency fix-ups.
            draft.cycle(key,delta); dirty=true;
            status="Menu preview / save; HUD on next launch";
            return;
        }
        edit(key,delta);
    }
    void refreshLearned() {
        learnedPaths={draft.values.at("net_profile")};
        fs::path parent=fs::path(draft.values.at("net_profile")).parent_path();
        if (fs::is_directory(parent)) for (auto &file : fs::directory_iterator(parent)) {
            if (learnedPaths.size()>=256) break;
            if (file.is_regular_file() && !file.is_symlink() && file.path().extension()==".bin") {
                try { validateLearned(file.path()); }
                catch (const std::exception &) { continue; }
                learnedPaths.push_back(file.path().string());
            }
        }
        std::sort(learnedPaths.begin(),learnedPaths.end());
        learnedPaths.erase(std::unique(learnedPaths.begin(),learnedPaths.end()),learnedPaths.end());
    }
    void cycleLearned(int delta) {
        auto reason=draft.disabledReason("net_profile",context);
        if (!reason.empty()) { status=reason; return; }
        refreshLearned();
        int at=int(std::find(learnedPaths.begin(),learnedPaths.end(),draft.values.at("net_profile"))-learnedPaths.begin());
        auto next=learnedPaths[(at+int(learnedPaths.size())+delta)%int(learnedPaths.size())];
        validateLearned(next);
        draft.set("net_profile",next);
        dirty=true;
        status="Unsaved / learned file unchanged";
    }
    void refresh() {
        names={"default",name};
        if (context.supportsPipeline("spatial")) names.push_back("spatial-experimental");
        if (fs::is_directory(directory)) for (auto &file : fs::directory_iterator(directory))
            if (file.is_regular_file() && !file.is_symlink() && file.path().extension()==".ini" && identifier(file.path().stem().string()))
                names.push_back(file.path().stem().string());
        std::sort(names.begin(),names.end()); names.erase(std::unique(names.begin(),names.end()),names.end());
    }
    void select(int delta) {
        if (dirty) { status="Save changes before switching presets"; return; }
        int at=int(std::find(names.begin(),names.end(),name)-names.begin());
        auto next=names[(at+int(names.size())+delta)%int(names.size())];
        auto loaded=load(directory/(next+".ini"));
        if (next=="spatial-experimental" && !fs::exists(directory/(next+".ini")))
            loaded.set("net_profile",draft.values.at("net_profile"));
        name=next; draft=loaded;
        if (draft.values["net_profile"].empty()) draft.set("net_profile",(directory/"learned.bin").string());
        refreshLearned(); status="Editing "+name+" / next launch";
    }
    void persist() {
        draft.validate(context);
        save(directory/(name+".ini"),draft);
        writeAtomic(directory/".active",name+"\n");
        dirty=false; status="Saved "+name+" / next launch"; refresh();
    }
    void clone() {
        draft.validate(context);
        for (int n=1;n<=999;++n) {
            auto next="custom-"+std::to_string(n);
            if (!fs::exists(directory/(next+".ini"))) {
                save(directory/(next+".ini"),draft);
                writeAtomic(directory/".active",next+"\n");
                name=next; dirty=false; refresh(); status="Created "+name+" / next launch"; return;
            }
        }
        throw std::runtime_error("Too many presets");
    }
};
} // namespace vapor

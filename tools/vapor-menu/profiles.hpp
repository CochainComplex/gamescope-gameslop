#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

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
    {"bidir", {"off", "on"}},
    {"hud", {"1", "2", "0"}},
    {"corner", {"br", "bl", "tr", "tl", "bc"}},
};
inline bool identifier(const std::string &s) {
    return !s.empty() && s.size()<=64 && s!="." && s!=".." &&
        std::all_of(s.begin(),s.end(),[](unsigned char c) {
            return (c>='a' && c<='z') || (c>='A' && c<='Z') ||
                (c>='0' && c<='9') || c=='-' || c=='_' || c=='.';
        });
}
struct Profile {
    std::map<std::string,std::string> values{
        {"framegen","on"}, {"mode","motion"}, {"pipeline","learned"},
        {"multiplier","2"}, {"online","on"}, {"bidir","off"},
        {"hud","1"}, {"corner","br"}, {"net_profile",""},
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
};
inline Profile load(const fs::path &path) {
    Profile p;
    if (!fs::exists(path)) return p;
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
    // Validate even when the caller constructed values directly.
    Profile checked;
    for (auto &[key,value] : p.values) checked.set(key,value);
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
    bool dirty=false;
    static std::string environment(const char *key, const std::string &fallback) {
        auto value=std::getenv(key); return value && *value ? value : fallback;
    }
    Store() {
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
    void refreshLearned() {
        learnedPaths={draft.values.at("net_profile")};
        fs::path parent=fs::path(draft.values.at("net_profile")).parent_path();
        if (fs::is_directory(parent)) for (auto &file : fs::directory_iterator(parent)) {
            if (learnedPaths.size()>=256) break;
            if (file.is_regular_file() && !file.is_symlink() && file.path().extension()==".bin")
                learnedPaths.push_back(file.path().string());
        }
        std::sort(learnedPaths.begin(),learnedPaths.end());
        learnedPaths.erase(std::unique(learnedPaths.begin(),learnedPaths.end()),learnedPaths.end());
    }
    void cycleLearned(int delta) {
        refreshLearned();
        int at=int(std::find(learnedPaths.begin(),learnedPaths.end(),draft.values.at("net_profile"))-learnedPaths.begin());
        draft.set("net_profile",learnedPaths[(at+int(learnedPaths.size())+delta)%int(learnedPaths.size())]);
        dirty=true;
    }
    void refresh() {
        names={name};
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
        name=next; draft=loaded;
        if (draft.values["net_profile"].empty()) draft.set("net_profile",(directory/"learned.bin").string());
        refreshLearned(); status="Editing "+name+" / next launch";
    }
    void persist() {
        save(directory/(name+".ini"),draft);
        writeAtomic(directory/".active",name+"\n");
        dirty=false; status="Saved "+name+" / next launch"; refresh();
    }
    void clone() {
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

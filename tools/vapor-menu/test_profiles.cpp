#include "profiles.hpp"
#include <cassert>
#include <iostream>

template<class Fn> static bool rejects(Fn fn) {
    try { fn(); return false; } catch (const std::exception &) { return true; }
}
int main(int argc, char **argv) {
    assert(argc==2);
    vapor::fs::path root=argv[1];
    vapor::fs::create_directories(root);
    assert(!vapor::identifier("../weights"));
    assert(!vapor::identifier(".."));
    assert(vapor::identifier("farcry6-test"));
    vapor::Profile p;
    p.set("pipeline","guided"); p.set("multiplier","4");
    p.set("net_profile",(root/"keep.bin").string());
    std::ofstream(root/"keep.bin") << "untouched learned weights";
    vapor::save(root/"roundtrip.ini",p);
    assert(vapor::load(root/"roundtrip.ini").values==p.values);
    assert(rejects([&]{p.set("multiplier","9");}));
    assert(rejects([&]{p.set("pipeline","shell $(command)");}));
    assert(rejects([&]{p.set("net_profile","weights\nhud=0");}));
    std::ofstream(root/"bad.ini") << "version=1\nmultiplier=2\nmultiplier=4\n";
    assert(rejects([&]{vapor::load(root/"bad.ini");}));
    std::ofstream(root/"bad.ini") << "version=2\n";
    assert(rejects([&]{vapor::load(root/"bad.ini");}));
    std::ofstream(root/"bad.ini") << std::string(9000,'x');
    assert(rejects([&]{vapor::load(root/"bad.ini");}));
    p.values["multiplier"]="9";
    assert(rejects([&]{vapor::save(root/"roundtrip.ini",p);}));
    assert(vapor::load(root/"roundtrip.ini").values.at("multiplier")=="4");
    setenv("VAPOR_PROFILE_ROOT",root.c_str(),1);
    setenv("VAPOR_GAME","test-game",1);
    setenv("VAPOR_PROFILE","default",1);
    vapor::Store store;
    store.draft.set("pipeline","warp"); store.dirty=true;
    store.persist(); store.clone();
    assert(store.name=="custom-1");
    store.draft.set("pipeline","learned"); store.dirty=true;
    store.select(-1); assert(store.name=="custom-1"); // Never lose edits.
    store.persist(); store.select(1);
    assert(store.name=="default");
    assert(store.draft.values.at("pipeline")=="warp");
    assert(vapor::active(root/"test-game")=="custom-1");
    // Explicit selection must not consult a broken last-used marker.
    std::ofstream(root/"test-game/.active") << "bad marker\nextra\n";
    setenv("VAPOR_PROFILE","default",1);
    vapor::Profile spatial=store.draft;
    spatial.set("pipeline","spatial");
    spatial.set("net_profile",(root/"keep.bin").string());
    vapor::save(root/"test-game/default.ini",spatial);
    const auto actual=(root/"override.bin").string();
    setenv("GAMESCOPE_FRAMEGEN_NET_PROFILE",actual.c_str(),1);
    vapor::Store overridden;
    assert(overridden.draft.values.at("pipeline")=="spatial");
    assert(overridden.draft.values.at("net_profile")==actual);
    assert(overridden.dirty);
    assert(!vapor::fs::exists(root/"override.bin"));
    overridden.persist();
    assert(vapor::load(root/"test-game/default.ini").values.at("net_profile")==actual);
    unsetenv("GAMESCOPE_FRAMEGEN_NET_PROFILE");
    std::ofstream(root/"test-game/.active") << "default\nextra\n";
    assert(rejects([&]{vapor::active(root/"test-game");}));
    std::ofstream(root/"test-game/.active") << "default\n";
    std::string weights; std::getline(std::ifstream(root/"keep.bin"),weights);
    assert(weights=="untouched learned weights");
    std::cout << "Profile persistence, validation, draft ownership and weight isolation PASS\n";
}

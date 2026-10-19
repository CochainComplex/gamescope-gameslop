#include "profiles.hpp"
#include <cassert>
#include <iostream>

template<class Fn> static bool rejects(Fn fn) {
    try { fn(); return false; } catch (const std::exception &) { return true; }
}
int main(int argc, char **argv) {
    assert(argc==2);
    if (std::string(argv[1])=="--matrix") {
        for (auto &fg : vapor::choices.at("framegen"))
        for (auto &mode : vapor::choices.at("mode"))
        for (auto &pipeline : vapor::choices.at("pipeline"))
        for (auto &bidir : vapor::choices.at("bidir"))
        for (auto backend : {"drm","wayland","sdl","headless","openvr"})
        for (bool base : {false,true}) for (bool vrr : {false,true}) {
            vapor::Profile p; p.set("framegen",fg); p.set("mode",mode);
            p.set("pipeline",pipeline); p.set("bidir",bidir);
            vapor::Context context{backend,base,vrr};
            std::cout << fg << ' ' << mode << ' ' << pipeline << ' ' << bidir << ' '
                << backend << ' ' << base << ' ' << vrr << ' '
                << !rejects([&]{p.validate(context);}) << ' ' << p.usesNet() << '\n';
        }
        return 0;
    }
    if (std::string(argv[1])=="--choices") {
        for (auto &[key,values] : vapor::choices) {
            std::cout << key;
            for (auto &value : values) std::cout << ' ' << value;
            std::cout << '\n';
        }
        return 0;
    }
    vapor::fs::path root=argv[1];
    vapor::fs::create_directories(root);
    const auto builtIn = vapor::load(root/"spatial-experimental.ini");
    assert(builtIn.values.at("pipeline")=="spatial");
    assert(builtIn.values.at("bidir")=="off");
    assert(builtIn.values.at("training_device")=="gpu");
    assert(!vapor::fs::exists(root/"spatial-experimental.ini"));
    auto unavailable=builtIn; unavailable.set("training_device","npu");
    assert(rejects([&]{vapor::save(root/"npu.ini",unavailable);}));
    assert(!vapor::fs::exists(root/"npu.ini"));

    assert(!vapor::identifier("../weights"));
    assert(!vapor::identifier(".."));
    assert(vapor::identifier("farcry6-test"));
    vapor::Profile p;
    p.set("pipeline","guided"); p.set("multiplier","4"); p.set("style","cga"); p.set("hud_opacity","70"); p.set("hud_scale","3");
    p.set("net_profile",(root/"keep.bin").string());
    const std::vector<gamescope::framegen::NetProfileSet> sets{
        {0xf0u,std::vector<float>(k_uFramegenNetFloats,0.0f)},
        {0x100f0u,std::vector<float>(k_uFramegenNetFloats,0.25f)}};
    const auto weights=gamescope::framegen::encode_net_profile(sets);
    assert(weights);
    std::ofstream(root/"keep.bin",std::ios::binary).write(
        reinterpret_cast<const char *>(weights->data()),weights->size());
    vapor::validateLearned(root/"keep.bin");
    std::ofstream(root/"legacy.bin") << "GSFR old or malformed weights";
    assert(rejects([&]{vapor::validateLearned(root/"legacy.bin");}));
    vapor::save(root/"roundtrip.ini",p);
    assert(vapor::load(root/"roundtrip.ini").values==p.values);
    assert(rejects([&]{p.set("multiplier","9");}));
    assert(rejects([&]{p.set("style","unknown");}));
    assert(rejects([&]{p.set("hud_opacity","110");}));
    assert(rejects([&]{p.set("hud_opacity","-10");}));
    assert(rejects([&]{p.set("hud_scale","7");}));
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
    auto selectedWeights=store.draft.values.at("net_profile");
    for (size_t i=0;i<store.names.size() && store.name!="spatial-experimental";++i) store.select(1);
    assert(store.name=="spatial-experimental");
    assert(store.draft.values.at("pipeline")=="spatial");
    assert(store.draft.values.at("net_profile")==selectedWeights);
    assert(!vapor::fs::exists(root/"test-game/spatial-experimental.ini"));
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
    std::ifstream saved(root/"keep.bin",std::ios::binary);
    std::vector<uint8_t> after((std::istreambuf_iterator<char>(saved)),{});
    assert(after==*weights);
    // UI dependencies do not permit inert controls or contradictory requests.
    overridden.draft=vapor::Profile{}; overridden.dirty=false;
    overridden.draft.set("pipeline","warp");
    overridden.edit("online",1);
    assert(!overridden.dirty && !overridden.status.empty());
    overridden.draft.set("pipeline","guided"); overridden.draft.set("bidir","on");
    overridden.edit("pipeline",1); // Guided -> Spatial retires bidir explicitly.
    assert(overridden.draft.values.at("pipeline")=="spatial");
    assert(overridden.draft.values.at("bidir")=="off");
    overridden.draft.validate();
    overridden.draft.set("pipeline","learned"); overridden.draft.set("bidir","on");
    overridden.edit("mode",1);
    assert(overridden.draft.values.at("bidir")=="off");
    assert(!overridden.draft.disabledReason("pipeline").empty());
    overridden.draft=vapor::Profile{}; overridden.context.backend="sdl";
    overridden.edit("bidir",1);
    assert(overridden.draft.values.at("bidir")=="off");
    overridden.draft.set("bidir","on"); // A saved preset from a different host.
    assert(rejects([&]{overridden.persist();}));
    overridden.edit("bidir",1); // It must still be possible to turn it off.
    assert(overridden.draft.values.at("bidir")=="off");
    overridden.context=vapor::Context{};
    overridden.context.pipelines={"warp","checked","learned","predict","guided"};
    overridden.draft=vapor::Profile{}; overridden.draft.set("pipeline","guided");
    overridden.edit("pipeline",1); // Skip Spatial on an older selected host.
    assert(overridden.draft.values.at("pipeline")=="warp");
    overridden.draft.set("pipeline","spatial");
    assert(rejects([&]{overridden.draft.validate(overridden.context);}));
    overridden.draft.set("framegen","off"); overridden.edit("framegen",-1);
    assert(overridden.draft.values.at("pipeline")=="warp");
    overridden.context.orientation="left";
    assert(rejects([&]{overridden.draft.validate(overridden.context);}));
    overridden.context.orientation="normal"; overridden.context.outputs="PerWindow";
    assert(rejects([&]{overridden.draft.validate(overridden.context);}));
    std::cout << "Profile persistence, validation, draft ownership and weight isolation PASS\n";
}

import os
from collections import namedtuple
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from matformer.matformer_registry import registry          
from matformer.plugins.socket_mixin import SocketMixin   
from matformer.plugins.plugin_classes import Plugin
from fnmatch import fnmatchcase
from matformer.plugins.plugin_classes import PluginConfig
from pathlib import Path
class PluginManager:
    def init_plugins(self):
        self.plugins = {}
        self.index_sockets()
        self.mount_configured()
        for target, plugin_cfg in getattr(self.model, 'default_plugins', lambda: [])():
            self.attach(target, plugin_cfg) #The default plugins
    def index_sockets(self):                   
        self.sockets = {}
        for path, module in self.model.named_modules():
            for name, socket in getattr(module, "sockets", {}).items():
                socket.address = f"{path}.{name}".lstrip(".")
                self.sockets[socket.address] = socket

    def attach(self, pattern, plugin, config=None, at=None, name=None):
        sockets = self.find_socket(pattern)
        if not sockets:
            raise ValueError(f"No socket matches attach pattern '{pattern}'. The pattern is wrong, or the mounted "
                             f"kernels don't expose it (e.g. a non-interpretable MLP kernel has no 'hidden_act' socket: "
                             f"check the MATFORMER_MLP_* env vars). Available sockets (first 15): {sorted(self.sockets)[:15]}")
        return [self._mount(s, plugin=plugin, index=at, name=name, config=config) for s in sockets.values()]
    
    def mount_from_files(self, path, position=None):
        """Restore plugins from files and mount them. If a folder or a list of folders is given, mount all the plugins in that folder (*.pt)"""
        path = [path] if not isinstance(path,list) else path
        files=[]
        return_dict={}
        for p in path:
            p = Path(p)
            if p.is_dir:
                files.extend([f for f in p.glob("*.pt")])
            else:
                files.append(p)
        for f in files:
            ckpt=torch.load(f, weights_only=False)
            if "plugin_kind" not in ckpt.keys():
                return_dict[f]="ERROR: File not recognized"
            else:
                sock=self.get_socket(ckpt['address'])
                if not sock:
                    return_dict[f]="ERROR: Wrong address"
                position = position if position else ckpt.get("index",-1)
                plugin=registry.create("plugin",ckpt['plugin_kind'],config=ckpt['plugin_config'], model_config=self.config, cache=self.cache)
                m,u=plugin.load(sd=ckpt['state_dict'])
                assert m+u==0,"Broken plugin file: missing or unexpected keys; please inspect manually"
                return_dict[f]=self._mount(sock, plugin, position, ckpt['plugin_kind'], ckpt['plugin_config'])
            
    
    def _mount(self, socket, plugin, index, name, config=None):
        if isinstance(plugin, PluginConfig) or (isinstance(plugin, type) and issubclass(plugin, PluginConfig)): #If it's a PluginConfig, it's instantiated     
            plugin = plugin.instantiate(use_cache=False, model_config=self.config, cache=self.cache)
        elif isinstance(plugin, type) and issubclass(plugin, Plugin):
            plugin = plugin(config=config, model_config=self.config, cache=self.cache)
        elif isinstance(plugin, type) and issubclass(plugin, nn.Module):
            plugin = plugin()
        elif not isinstance(plugin, nn.Module):    
            raise Exception(f"Unmountable plugin: {plugin}")
        kind = name or getattr(plugin, "name", None)
        assert kind is not None,f"The plugin misses its kind. Please provide a name to the _mount function."
        plugin.kind=kind
        taken = {getattr(component, "name", None) for component in socket.components}
        plugin.name = plugin.kind
        n=0
        while plugin.name in taken: #Rename if a plugin with same name exists                 
            n += 1
            plugin.name = f"{kind}_{n}"

        socket.add(plugin, index)
        plugin.socket = socket
        plugin.id = f"{socket.address}_{plugin.name}"
        if isinstance(plugin, nn.Module): # added to the model's state dict
            plugin._key = f"{socket.name}_{plugin.name}"            
            socket.owner.add_module(plugin._key, plugin)
            socket.owner.__dict__.setdefault("__instance_annotations__", {})[plugin._key] = \
                f"param_name:{socket.name}.{plugin.kind}"      # keeps freeze-by-stable-name working
        self.plugins[plugin.id]=plugin 
        return plugin

    def mount_configured(self):
        if hasattr(self.configs,"plugin"):
            for plugin_cfg in self.configs.plugin.instances():
                print(plugin_cfg)
                targets=plugin_cfg.targets
                if targets is None:
                    continue #Plugin declared but not auto-mounted
                targets=[targets] if not isinstance(targets,list) else targets
                for target in targets:
                    self.attach(target, plugin_cfg, name=plugin_cfg.name)


    def find_socket(self, pattern="*"):
        if not hasattr(self,"sockets"):
            raise ValueError(f"Plugin Manager not initialised yet. Run init_plugins()")
        patterns = [pattern] if isinstance(pattern, str) else pattern
        found = {}
        for address, socket in self.sockets.items():
            if any(fnmatchcase(address, pattern) for pattern in patterns):
                found[address] = socket
        return found
    
    def get_socket(self,socket_name):
        return self.sockets.get(socket_name)

    def find_plugin(self, pattern="*"):
        patterns = [pattern] if isinstance(pattern, str) else pattern
        found = []        
        for plugin in self.plugins.values():
            if any(fnmatchcase(plugin.id, p) for p in patterns):
                found.append(plugin) 
        return found[0] if len(found)==1 else found

    def add_first(self, pattern, plugin, config=None, at=None, name=None): return self.attach(pattern=pattern, plugin=plugin, config=config, at=0, name=name)
    def add_end(self, pattern, plugin, config=None, at=None, name=None):   return self.attach(pattern=pattern, plugin=plugin, config=config, at=-1, name=name)

    def detach(self, pattern, ref):
        for s in self.find_socket(pattern).values():
            self._unmount(s, ref)

    def replace(self, pattern, plugin, ref=0):
        for s in self.find_socket(pattern).values():
            i = s.index(ref)
            self._mount(s, plugin, i, None)          # new one goes in front of the old one
            self._unmount(s, i + 1)

    def swap(self, pattern, a, b):
        for s in self.find_socket(pattern).values(): s.swap(a, b)

    def reorder(self, pattern, order):
        for s in self.find_socket(pattern).values(): s.reorder(order)

    def inspect(self, pattern="*"):
        return {a: s.inspect() for a, s in self.find_socket(pattern).items()}

    def all_plugin_params(self):
        params = []
        for p in self.plugins.values():
            if hasattr(p, "parameters"):
                params.extend(p.parameters())
        return params

    def reset_visits(self):                          # once per forward
        for m in self.model.modules():
            for s in getattr(m, "sockets", {}).values():
                s.visits = 0

    def _unmount(self, sock, ref):
        plugin=None
        try:
            plugin = sock.remove(ref)
            if getattr(plugin, "_key", None):
                delattr(sock.owner, plugin._key)
                sock.owner.__dict__.get("__instance_annotations__", {}).pop(plugin._key, None)
            self.plugins.pop(getattr(plugin, "id", None), None)
        except IndexError:
            print(f"No plugins at socket {sock}")
        return plugin

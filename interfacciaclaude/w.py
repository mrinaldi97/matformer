"""LLM-Lab — wxPython structural stub.

Mimics the PyQt LLM-Lab shell: menubar + toolbar + navigator tree + notebook
+ plugin form + jobs table + status LEDs.  All backend calls are stubs.
"""
import json
import time
import wx


class Main(wx.Frame):
    def __init__(self):
        super().__init__(None, title="LLM-Lab (wxPython stub)", size=(1250, 760))
        self._sash_nav, self._sash_form, self._sash_jobs = 200, 340, 180

        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self.Centre()

    # ── Menu ────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = wx.MenuBar()

        m_file = wx.Menu()
        self.mi_connect = m_file.Append(wx.ID_ANY, "&Connect\tCtrl+Shift+C")
        m_file.Append(wx.ID_SAVE, "&Save settings\tCtrl+S")
        m_file.Append(wx.ID_OPEN, "&Load settings\tCtrl+O")
        m_file.AppendSeparator()
        m_file.Append(wx.ID_EXIT, "E&xit\tCtrl+Q")
        mb.Append(m_file, "&File")

        m_view = wx.Menu()
        self.mi_nav = m_view.AppendCheckItem(wx.ID_ANY, "Toggle &Navigator\tCtrl+1")
        self.mi_nav.Check(True)
        self.mi_jobs = m_view.AppendCheckItem(wx.ID_ANY, "Toggle &Jobs\tCtrl+2")
        self.mi_jobs.Check(True)
        self.mi_form = m_view.AppendCheckItem(wx.ID_ANY, "Toggle &Form\tCtrl+3")
        self.mi_form.Check(True)
        mb.Append(m_view, "&View")

        m_help = wx.Menu()
        m_help.Append(wx.ID_ABOUT, "&About")
        mb.Append(m_help, "&Help")

        self.SetMenuBar(mb)

        self.Bind(wx.EVT_MENU, lambda e: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self._on_about,       id=wx.ID_ABOUT)
        self.Bind(wx.EVT_MENU, self._on_save,        id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self._on_load,        id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self._on_connect,     self.mi_connect)
        self.Bind(wx.EVT_MENU, self._on_toggle_nav,  self.mi_nav)
        self.Bind(wx.EVT_MENU, self._on_toggle_jobs, self.mi_jobs)
        self.Bind(wx.EVT_MENU, self._on_toggle_form, self.mi_form)

    # ── Body ────────────────────────────────────────────────────────────
    def _build_body(self):
        # Build splitters top-down so each pane can be parented to its splitter.
        # Nesting: self → outer_split → mid_split → center_split
        self.outer_split  = wx.SplitterWindow(self,             style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self.mid_split    = wx.SplitterWindow(self.outer_split, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self.center_split = wx.SplitterWindow(self.mid_split,   style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)

        # Build panes with the correct parent.
        self.toolbar_panel = self._build_toolbar(self)
        self.nav_panel     = self._build_nav(self.outer_split)
        self.notebook      = self._build_notebook(self.center_split)
        self.jobs_panel    = self._build_jobs(self.center_split)
        self.form_panel    = self._build_form(self.mid_split)

        # Notebook over Jobs (vertical)
        self.center_split.SplitHorizontally(self.notebook, self.jobs_panel, -self._sash_jobs)
        self.center_split.SetMinimumPaneSize(80)

        # Center over Form (horizontal)
        self.mid_split.SplitVertically(self.center_split, self.form_panel, -self._sash_form)
        self.mid_split.SetMinimumPaneSize(150)

        # Navigator over the rest (horizontal)
        self.outer_split.SplitVertically(self.nav_panel, self.mid_split, self._sash_nav)
        self.outer_split.SetMinimumPaneSize(120)

        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(self.toolbar_panel, 0, wx.EXPAND)
        s.Add(self.outer_split,   1, wx.EXPAND)
        self.SetSizer(s)

    # ── Toolbar ─────────────────────────────────────────────────────────
    def _build_toolbar(self, parent):
        p = wx.Panel(parent)
        p.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE))
        s = wx.BoxSizer(wx.HORIZONTAL)
        for label, cb in [("Connect", self._on_connect),
                          ("Send",    self._on_send),
                          ("Stop",    self._on_stop),
                          ("Clear",   self._on_clear)]:
            b = wx.Button(p, label=label)
            b.Bind(wx.EVT_BUTTON, cb)
            s.Add(b, 0, wx.ALL, 4)
        p.SetSizer(s)
        return p

    # ── Navigator tree ──────────────────────────────────────────────────
    def _build_nav(self, parent):
        p = wx.Panel(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(p, label="Navigator")
        t.SetFont(t.GetFont().Bold())
        s.Add(t, 0, wx.ALL, 5)

        self.tree = wx.TreeCtrl(p, style=wx.TR_HAS_BUTTONS | wx.TR_LINES_AT_ROOT | wx.TR_HIDE_ROOT)
        root   = self.tree.AddRoot("root")
        work   = self.tree.AppendItem(root, "Work")
        for name in ("Chat", "Models", "Datasets", "Training"):
            self.tree.AppendItem(work, name)
        system = self.tree.AppendItem(root, "System")
        for name in ("Jobs", "Config"):
            self.tree.AppendItem(system, name)
        self.tree.ExpandAll()
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self._on_tree_sel)

        s.Add(self.tree, 1, wx.EXPAND | wx.ALL, 5)
        p.SetSizer(s)
        return p

    # ── Notebook (Chat / Tables / Log) ──────────────────────────────────
    def _build_notebook(self, parent):
        nb = wx.Notebook(parent)

        # Chat
        chat = wx.Panel(nb)
        s = wx.BoxSizer(wx.VERTICAL)
        top = wx.BoxSizer(wx.HORIZONTAL)
        top.Add(wx.StaticText(chat, label="Run on server:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        self.server_choice = wx.Choice(chat, choices=["server1", "server2"])
        self.server_choice.SetSelection(0)
        top.Add(self.server_choice, 0, wx.ALL, 4)
        s.Add(top, 0, wx.EXPAND)

        self.transcript = wx.TextCtrl(chat, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.transcript.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE,
                                        wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        s.Add(self.transcript, 1, wx.EXPAND | wx.ALL, 4)

        self.prompt = wx.TextCtrl(chat, style=wx.TE_MULTILINE, size=(-1, 80))
        self.prompt.SetHint("Prompt…  Ctrl+Enter sends, Esc stops")
        self.prompt.Bind(wx.EVT_KEY_DOWN, self._on_prompt_key)
        s.Add(self.prompt, 0, wx.EXPAND | wx.ALL, 4)
        chat.SetSizer(s)
        nb.AddPage(chat, "Chat")

        # Tables
        tables = wx.Panel(nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.tables_list = wx.ListCtrl(tables, style=wx.LC_REPORT)
        self.tables_list.InsertColumn(0, "Server", width=100)
        self.tables_list.InsertColumn(1, "Name",   width=220)
        self.tables_list.InsertColumn(2, "Size",   width=100)
        for i, (sv, nm, sz) in enumerate([("server1", "llama-3-8b", "16 GB"),
                                          ("server1", "mistral-7b", "14 GB"),
                                          ("server2", "phi-3",      " 7 GB")]):
            self.tables_list.InsertItem(i, sv)
            self.tables_list.SetItem(i, 1, nm)
            self.tables_list.SetItem(i, 2, sz)
        s.Add(self.tables_list, 1, wx.EXPAND | wx.ALL, 4)
        tables.SetSizer(s)
        nb.AddPage(tables, "Tables")

        # Log
        log = wx.Panel(nb)
        s = wx.BoxSizer(wx.VERTICAL)
        self.log_txt = wx.TextCtrl(log, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.log_txt.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE,
                                     wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        s.Add(self.log_txt, 1, wx.EXPAND | wx.ALL, 4)
        log.SetSizer(s)
        nb.AddPage(log, "Log")

        nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self._on_tab_changed)
        return nb

    # ── Jobs table ──────────────────────────────────────────────────────
    def _build_jobs(self, parent):
        p = wx.Panel(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(p, label="Jobs — all servers")
        t.SetFont(t.GetFont().Bold())
        s.Add(t, 0, wx.ALL, 3)

        self.jobs_list = wx.ListCtrl(p, style=wx.LC_REPORT)
        headers = ["Server", "ID", "Kind", "Name", "Status", "Progress"]
        for i, h in enumerate(headers):
            self.jobs_list.InsertColumn(i, h, width=140 if h == "Name" else 90)
        for i, row in enumerate([
            ("server1", "job-1", "train", "run-001", "running", " 45%"),
            ("server2", "job-2", "eval",  "run-002", "done",    "100%"),
        ]):
            self.jobs_list.InsertItem(i, row[0])
            for j, v in enumerate(row[1:], 1):
                self.jobs_list.SetItem(i, j, v)
        s.Add(self.jobs_list, 1, wx.EXPAND | wx.ALL, 4)
        p.SetSizer(s)
        return p

    # ── Plugin form (right) ─────────────────────────────────────────────
    def _build_form(self, parent):
        p = wx.Panel(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(p, label="Plugin Form")
        t.SetFont(t.GetFont().Bold())
        s.Add(t, 0, wx.ALL, 5)

        form = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        form.AddGrowableCol(1, 1)

        form.Add(wx.StaticText(p, label="Temperature:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_temp = wx.SpinCtrlDouble(p, min=0.0, max=2.0, inc=0.05, initial=0.8)
        self.spin_temp.SetDigits(2)
        form.Add(self.spin_temp, 1, wx.EXPAND)

        form.Add(wx.StaticText(p, label="Max tokens:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_max = wx.SpinCtrl(p, min=1, max=100000, initial=512)
        form.Add(self.spin_max, 1, wx.EXPAND)

        form.Add(wx.StaticText(p, label="Model:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.model_choice = wx.Choice(p, choices=["llama-3-8b", "mistral-7b", "phi-3"])
        self.model_choice.SetSelection(0)
        form.Add(self.model_choice, 1, wx.EXPAND)

        form.Add(wx.StaticText(p, label="Path:"), 0, wx.ALIGN_CENTER_VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.path_edit = wx.TextCtrl(p, value="/models/")
        row.Add(self.path_edit, 1, wx.EXPAND)
        browse = wx.Button(p, label="…", size=(30, -1))
        browse.Bind(wx.EVT_BUTTON, self._on_browse)
        row.Add(browse, 0)
        form.Add(row, 1, wx.EXPAND)

        form.Add(wx.StaticText(p, label="Highlight:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.chk_highlight = wx.CheckBox(p)
        self.chk_highlight.SetValue(True)
        form.Add(self.chk_highlight, 0)

        s.Add(form, 0, wx.EXPAND | wx.ALL, 6)

        apply_btn = wx.Button(p, label="Apply")
        apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
        s.Add(apply_btn, 0, wx.ALL, 6)
        s.AddStretchSpacer(1)
        p.SetSizer(s)
        return p

    # ── Status bar with LEDs ────────────────────────────────────────────
    def _build_statusbar(self):
        p = wx.Panel(self)
        s = wx.BoxSizer(wx.HORIZONTAL)
        self.status_text = wx.StaticText(p, label="Ready.")
        s.Add(self.status_text, 1, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)

        self.leds = {}
        for name in ("server1", "server2"):
            lbl = wx.StaticText(p, label=f"\u25cf {name}  ")
            self.leds[name] = lbl
            self._paint_led(name, False)
            s.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 6)
        p.SetSizer(s)

        self.GetSizer().Add(p, 0, wx.EXPAND)
        self.Layout()

    # ── Small helpers ───────────────────────────────────────────────────
    def _log(self, msg):
        self.log_txt.AppendText(f"{time.strftime('%H:%M:%S')}  {msg}\n")

    def _set_status(self, msg):
        self.status_text.SetLabel(msg)

    def _paint_led(self, name, up):
        c = wx.Colour(0x2a, 0xaa, 0x2a) if up else wx.Colour(0xcc, 0x22, 0x22)
        self.leds[name].SetForegroundColour(c)
        self.leds[name].Refresh()

    # ── Callbacks: menu / toolbar ───────────────────────────────────────
    def _on_about(self, e):
        wx.MessageBox(
            "LLM-Lab wxPython stub.\n"
            "Structural mimic of the PyQt shell, built with wxWidgets.",
            "About", wx.OK | wx.ICON_INFORMATION, self)

    def _on_connect(self, e):
        self._set_status("Connecting…")
        for n in self.leds:
            self._paint_led(n, True)
            self._log(f"{n}: connected")

    def _on_save(self, e):
        with wx.FileDialog(self, "Save settings",
                           wildcard="JSON (*.json)|*.json",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                data = {"temperature": self.spin_temp.GetValue(),
                        "max_tokens":  self.spin_max.GetValue(),
                        "model":       self.model_choice.GetStringSelection(),
                        "path":        self.path_edit.GetValue(),
                        "highlight":   self.chk_highlight.GetValue()}
                with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._log(f"saved settings → {dlg.GetPath()}")

    def _on_load(self, e):
        with wx.FileDialog(self, "Open settings",
                           wildcard="JSON (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                with open(dlg.GetPath(), encoding="utf-8") as f:
                    d = json.load(f)
                if "temperature" in d: self.spin_temp.SetValue(d["temperature"])
                if "max_tokens"  in d: self.spin_max.SetValue(d["max_tokens"])
                if "model"       in d: self.model_choice.SetStringSelection(d["model"])
                if "path"        in d: self.path_edit.SetValue(d["path"])
                if "highlight"   in d: self.chk_highlight.SetValue(bool(d["highlight"]))
                self._log(f"loaded settings ← {dlg.GetPath()}")

    # ── Callbacks: view toggles ─────────────────────────────────────────
    def _on_toggle_nav(self, e):
        if self.mi_nav.IsChecked():
            if not self.outer_split.IsSplit():
                self.outer_split.SplitVertically(self.nav_panel, self.mid_split, self._sash_nav)
        else:
            if self.outer_split.IsSplit():
                self._sash_nav = self.outer_split.GetSashPosition()
                self.outer_split.Unsplit(self.nav_panel)

    def _on_toggle_jobs(self, e):
        if self.mi_jobs.IsChecked():
            if not self.center_split.IsSplit():
                self.center_split.SplitHorizontally(self.notebook, self.jobs_panel, -self._sash_jobs)
        else:
            if self.center_split.IsSplit():
                self._sash_jobs = self.center_split.GetSize().height - self.center_split.GetSashPosition()
                self.center_split.Unsplit(self.jobs_panel)

    def _on_toggle_form(self, e):
        if self.mi_form.IsChecked():
            if not self.mid_split.IsSplit():
                self.mid_split.SplitVertically(self.center_split, self.form_panel, -self._sash_form)
        else:
            if self.mid_split.IsSplit():
                self._sash_form = self.mid_split.GetSize().width - self.mid_split.GetSashPosition()
                self.mid_split.Unsplit(self.form_panel)

    # ── Callbacks: chat / tree / form ───────────────────────────────────
    def _on_tree_sel(self, e):
        item = e.GetItem()
        if item.IsOk():
            self._set_status(f"Navigator: {self.tree.GetItemText(item)}")

    def _on_tab_changed(self, e):
        i = self.notebook.GetSelection()
        self._set_status(f"Tab: {self.notebook.GetPageText(i)}")

    def _on_prompt_key(self, e):
        if e.GetKeyCode() == wx.WXK_RETURN and e.ControlDown():
            self._on_send(None)
        else:
            e.Skip()

    def _on_send(self, e):
        text = self.prompt.GetValue().strip()
        if not text:
            return
        self.transcript.AppendText(f"you> {text}\n")
        self.prompt.Clear()
        self._set_status("Generating…")
        wx.CallLater(400, self._fake_reply)

    def _fake_reply(self):
        self.transcript.AppendText("model> (stub) hello from wxPython demo.\n\n")
        self._set_status("Done.")

    def _on_stop(self, e):
        self._set_status("Stopped.")

    def _on_clear(self, e):
        self.transcript.Clear()
        self._set_status("Cleared.")

    def _on_apply(self, e):
        vals = {"temperature": self.spin_temp.GetValue(),
                "max_tokens":  self.spin_max.GetValue(),
                "model":       self.model_choice.GetStringSelection(),
                "path":        self.path_edit.GetValue(),
                "highlight":   self.chk_highlight.GetValue()}
        self._log(f"apply → {vals}")
        self._set_status("Form applied to server.")

    def _on_browse(self, e):
        with wx.DirDialog(self, "Pick a directory") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.path_edit.SetValue(dlg.GetPath())


class App(wx.App):
    def OnInit(self):
        Main().Show()
        return True


if __name__ == "__main__":
    App().MainLoop()

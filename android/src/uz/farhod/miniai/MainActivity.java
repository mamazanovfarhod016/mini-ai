package uz.farhod.miniai;

import android.app.Activity;
import android.content.Intent;
import android.content.res.Configuration;
import android.net.Uri;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/**
 * Mini AI: обёртка WebView. Интерфейс и нейросеть (model.js + brain.js) лежат в assets/www,
 * вся «думалка» — на JavaScript прямо в телефоне. Интернет — для самообучения (сервер Mini AI,
 * поиск ответа в Википедии); ссылки на источники открываются во внешнем браузере.
 */
public class MainActivity extends Activity {
    static final String URL = "file:///android_asset/www/index.html";
    static final int LIGHT = Color.parseColor("#F5F4EE");
    static final int DARK = Color.parseColor("#262624");
    WebView web;

    boolean systemDark() {
        return (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
    }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        web = new WebView(this);
        applyBars(systemDark());
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // localStorage — история чатов
        s.setAllowFileAccess(true);
        s.setAllowContentAccess(false);
        s.setAllowUniversalAccessFromFileURLs(false);
        s.setTextZoom(100);
        web.addJavascriptInterface(new Bridge(), "AndroidApp");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("file".equals(u.getScheme())) return false;            // свои страницы — внутри
                if ("https".equals(u.getScheme())) {                       // источник (Википедия) — в браузере
                    try { startActivity(new Intent(Intent.ACTION_VIEW, u)); } catch (Exception ignored) {}
                }
                return true;
            }
        });
        setContentView(web);
        if (b != null) web.restoreState(b); else web.loadUrl(URL);
    }

    void applyBars(boolean dark) {
        Window w = getWindow();
        int c = dark ? DARK : LIGHT;
        w.setStatusBarColor(c);
        w.setNavigationBarColor(c);
        View d = w.getDecorView();
        int f = d.getSystemUiVisibility();
        if (Build.VERSION.SDK_INT >= 23) {
            if (dark) f &= ~View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR; else f |= View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
        }
        if (Build.VERSION.SDK_INT >= 26) {
            if (dark) f &= ~View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR; else f |= View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
        }
        d.setSystemUiVisibility(f);
        if (web != null) web.setBackgroundColor(c);
    }

    class Bridge {
        @JavascriptInterface public void setTheme(final boolean dark) { runOnUiThread(() -> applyBars(dark)); }
    }

    @Override public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        applyBars(systemDark());
    }
    @Override protected void onSaveInstanceState(Bundle o) { super.onSaveInstanceState(o); web.saveState(o); }
    @Override public void onBackPressed() {
        web.evaluateJavascript("window.miniBack?window.miniBack():0", v -> {
            if (!"1".equals(v)) MainActivity.super.onBackPressed();
        });
    }
    @Override protected void onDestroy() { web.destroy(); super.onDestroy(); }
}
